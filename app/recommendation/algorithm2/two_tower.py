"""C0: content-driven two-tower scorer for algorithm 2 (PyTorch).

collection tower  input = profile of the visible collection H: tag frequency vectors (genre, theme,
                  demographic, studio), known-field fractions, median visible episodes (log scale) and
                  its missing mask, log |H|, mean catalogue score of H and its mask
item tower        input = the candidate's own metadata: genre / theme / demographic / studio multi-hot,
                  catalogue score / 10, log episodes, and a missing mask per field
score             dot product of the two output vectors (a raw logit; trained with BCEWithLogitsLoss)

No user-id or anime-id embedding is used, so a new collection or a title without interactions is
encoded from its metadata alone. Numeric inputs use fixed transforms (score / 10, log1p(x) / log1p(1000)),
so nothing is fitted on evaluation data. The studio vocabulary keeps studios with at least
`studio_min_count` catalogue titles (a metadata-only rule); the rest share one UNK column.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from app.recommendation.algorithm2.catalog import NAMESPACES, Alg2Catalog
from app.recommendation.algorithm2.profiles import Alg2Profile

ENCODER_VERSION = "alg2-tower-encoder-v1"
LOG_SCALE = float(np.log1p(1000.0))


class TowerEncoder:
    def __init__(self, cat: Alg2Catalog, studio_min_count: int = 5):
        self.cat = cat
        self.studio_min_count = int(studio_min_count)
        counts = np.asarray(cat.X["studio"][cat.members].sum(axis=0)).ravel()
        keep = np.flatnonzero(counts >= self.studio_min_count)
        self.studio_map = np.full(cat.X["studio"].shape[1], len(keep), dtype=np.int64)   # default -> UNK
        self.studio_map[keep] = np.arange(len(keep))
        self.n_studio = len(keep) + 1
        self.dims = {ns: cat.X[ns].shape[1] for ns in ("genre", "theme", "demographic")}
        self.dims["studio"] = self.n_studio

    # ------------------------------------------------------------------ spec / compatibility
    def spec(self) -> dict:
        vocab_hash = hashlib.sha256(json.dumps({ns: self.cat.vocab[ns] for ns in NAMESPACES},
                                               ensure_ascii=False).encode()).hexdigest()
        return {"version": ENCODER_VERSION, "studio_min_count": self.studio_min_count, "dims": self.dims,
                "vocab_sha256": vocab_hash, "d_item": self.d_item, "d_coll": self.d_coll}

    @property
    def d_item(self) -> int:
        return sum(self.dims.values()) + 2 + 5

    @property
    def d_coll(self) -> int:
        return sum(self.dims.values()) + 4 + 2 + 1 + 2

    def _studio_onehot(self, rows) -> np.ndarray:
        X = self.cat.X["studio"][rows]
        out = np.zeros((X.shape[0], self.n_studio), dtype=np.float32)
        r, c = X.nonzero()
        np.add.at(out, (r, self.studio_map[c]), 1.0)
        return np.minimum(out, 1.0)

    # ------------------------------------------------------------------ item tower input
    def item_matrix(self, rows: np.ndarray | None = None) -> np.ndarray:
        cat = self.cat
        rows = np.arange(cat.n) if rows is None else np.asarray(rows, dtype=np.int64)
        parts = [cat.X[ns][rows].toarray().astype(np.float32) for ns in ("genre", "theme", "demographic")]
        parts.append(self._studio_onehot(rows))
        score = cat.score[rows]
        eps = cat.episodes[rows]
        sc_ok, ep_ok = np.isfinite(score), np.isfinite(eps)
        num = np.column_stack([np.where(sc_ok, score / 10.0, 0.0),
                               np.where(ep_ok, np.log1p(np.where(ep_ok, eps, 0.0)) / LOG_SCALE, 0.0)])
        masks = np.column_stack([cat.sizes[ns][rows] == 0 for ns in NAMESPACES] + [~ep_ok])
        return np.hstack(parts + [num.astype(np.float32), masks.astype(np.float32)])

    # ------------------------------------------------------------------ collection tower input
    def collection_vector(self, prof: Alg2Profile) -> np.ndarray:
        parts = [prof.freq[ns].astype(np.float32) for ns in ("genre", "theme", "demographic")]
        studio = np.zeros(self.n_studio, dtype=np.float32)
        np.add.at(studio, self.studio_map, prof.freq["studio"].astype(np.float32))
        parts.append(studio)
        size = max(prof.size, 1)
        known = np.array([prof.known[ns] / size for ns in NAMESPACES], dtype=np.float32)
        med_ok = prof.median_eps is not None
        med = np.array([np.log1p(prof.median_eps) / LOG_SCALE if med_ok else 0.0, 0.0 if med_ok else 1.0],
                       dtype=np.float32)
        hist = np.array([np.log1p(prof.size) / LOG_SCALE], dtype=np.float32)
        sc_ok = prof.mean_score is not None
        sc = np.array([prof.mean_score / 10.0 if sc_ok else 0.0, 0.0 if sc_ok else 1.0], dtype=np.float32)
        return np.concatenate(parts + [known, med, hist, sc])


def _require_torch():
    import torch
    from torch import nn

    return torch, nn


def make_model(d_coll: int, d_item: int, out_dim: int, dropout: float, hidden: tuple[int, int] = (128, 64)):
    torch, nn = _require_torch()

    def tower(d_in: int):
        return nn.Sequential(nn.Linear(d_in, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden[1], out_dim))

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.coll = tower(d_coll)
            self.item = tower(d_item)

        def forward(self, c, i):
            return (self.coll(c) * self.item(i)).sum(dim=-1)

    return TwoTower()


def train_two_tower(coll_X: np.ndarray, item_X: np.ndarray, train: tuple, val: tuple, cfg: dict, seed: int,
                    device: str = "cpu", log=print):
    """train / val = (query_row, item_row, label) arrays indexing coll_X / item_X.

    Early stopping on the validation BCE loss (patience `early_stopping_patience`); the weights of the best
    epoch are restored. Returns (model, history).
    """
    torch, nn = _require_torch()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = make_model(coll_X.shape[1], item_X.shape[1], int(cfg["out_dim"]), float(cfg["dropout"]),
                       tuple(cfg.get("hidden", (128, 64)))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()
    C = torch.as_tensor(coll_X, dtype=torch.float32, device=device)
    Itm = torch.as_tensor(item_X, dtype=torch.float32, device=device)
    tq, ti, ty = (torch.as_tensor(np.asarray(a), device=device) for a in train)
    vq, vi, vy = (torch.as_tensor(np.asarray(a), device=device) for a in val)
    ty, vy = ty.float(), vy.float()
    bs = int(cfg.get("batch_size", 512))
    max_epochs, patience = int(cfg.get("max_epochs", 30)), int(cfg.get("early_stopping_patience", 3))
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "seconds": []}
    best, best_loss, bad = None, float("inf"), 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.as_tensor(np_rng.permutation(len(ty)), device=device)
        total, seen = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            logits = model(C[tq[b]], Itm[ti[b]])
            loss = loss_fn(logits, ty[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            seen += len(b)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, len(vy), 8192):
                vl += float(loss_fn(model(C[vq[s:s + 8192]], Itm[vi[s:s + 8192]]), vy[s:s + 8192])) * len(vy[s:s + 8192])
            vl /= max(1, len(vy))
        hist["epoch"].append(epoch)
        hist["train_loss"].append(round(total / max(1, seen), 6))
        hist["val_loss"].append(round(vl, 6))
        hist["seconds"].append(round(time.time() - t0, 1))
        if vl < best_loss - 1e-6:
            best_loss, bad = vl, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
        else:
            bad += 1
            if bad >= patience:
                hist["stopped"] = "early_stopping"
                break
    else:
        hist["stopped"] = "max_epochs"
    if best is not None:
        model.load_state_dict(best)
    hist["best_val_loss"] = round(best_loss, 6)
    hist["device"] = device
    model.eval()
    return model, hist


def item_embeddings(model, item_X: np.ndarray, device: str = "cpu") -> np.ndarray:
    torch, _ = _require_torch()
    with torch.no_grad():
        out = model.item(torch.as_tensor(item_X, dtype=torch.float32, device=device))
    return out.cpu().numpy().astype(np.float32)


def collection_embeddings(model, coll_X: np.ndarray, device: str = "cpu") -> np.ndarray:
    torch, _ = _require_torch()
    with torch.no_grad():
        out = model.coll(torch.as_tensor(np.atleast_2d(coll_X), dtype=torch.float32, device=device))
    return out.cpu().numpy().astype(np.float32)


def save_model(model, path, cfg: dict, encoder_spec: dict, extra: dict) -> None:
    torch, _ = _require_torch()
    torch.save({"state_dict": model.state_dict(), "config": cfg, "encoder": encoder_spec, "extra": extra}, path)


def load_model(path):
    torch, _ = _require_torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    enc = blob["encoder"]
    cfg = blob["config"]
    model = make_model(enc["d_coll"], enc["d_item"], int(cfg["out_dim"]), float(cfg["dropout"]),
                       tuple(cfg.get("hidden", (128, 64))))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
