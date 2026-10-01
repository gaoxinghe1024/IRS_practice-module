"""Algorithm 2 scorer C: two-tower model (collection tower x item content tower).

Both towers encode *content*: the collection tower pools the content vectors
of the visible collection items (so a brand-new user works), the item tower
encodes a title's own tags / length / score bucket (so a new title works).
Embeddings are L2-normalised; score = temperature * dot product. Trained with a
sampled-softmax loss over the hidden positive and negatives drawn from the same
rule-filtered pool.

It is only deployed when (a) the number of real collection interactions passes
``two_tower.min_interactions`` and (b) validation shows a gain; otherwise the
capability is reported as limited_by_data. Requires the optional ``ml`` extra.
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.profiles import CollectionProfile

SCORER_ID = "two_tower_v1"
EP_BUCKETS = [1, 2, 7, 14, 27, 60]
SCORE_BUCKETS = [6.0, 7.0, 7.5, 8.0, 8.5]


def item_content_matrix(snap: CatalogSnapshot, namespaces=("genre", "theme", "demographic", "studio")) -> np.ndarray:
    blocks = []
    for ns in namespaces:
        tm = snap.tags.get(ns)
        if tm is not None:
            blocks.append(tm.X.toarray().astype(np.float32))
    e = snap.episodes
    eb = np.zeros((snap.n, len(EP_BUCKETS) + 2), dtype=np.float32)
    known = np.isfinite(e)
    eb[~known, -1] = 1
    eb[known, np.searchsorted(EP_BUCKETS, e[known], side="left")] = 1
    s = snap.score
    sb = np.zeros((snap.n, len(SCORE_BUCKETS) + 2), dtype=np.float32)
    known = np.isfinite(s)
    sb[~known, -1] = 1
    sb[known, np.searchsorted(SCORE_BUCKETS, s[known], side="left")] = 1
    blocks += [eb, sb]
    return np.hstack(blocks)


def _net(n_feat: int, dim: int):
    from torch import nn

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.item = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Linear(128, dim))
            self.coll = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Linear(128, dim))

        def encode_items(self, x):
            return nn.functional.normalize(self.item(x), dim=-1)

        def encode_collection(self, x):
            return nn.functional.normalize(self.coll(x), dim=-1)

    return TwoTower()


class TwoTowerScorer:
    scorer_id = SCORER_ID
    version = "1.0"
    temperature = 10.0

    def __init__(self, net, content: np.ndarray, meta: dict):
        self.net = net
        self.content = content
        self.meta = meta

    def score(self, snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray
              ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        import torch

        with torch.no_grad():
            c = self.net.encode_collection(torch.as_tensor(self.content[profile.items].mean(0, keepdims=True)))
            it = self.net.encode_items(torch.as_tensor(self.content[candidates]))
            s = (it @ c.T).squeeze(1).numpy() * self.temperature
        return s, {"dot": s}

    def save(self, path) -> None:
        import torch

        torch.save({"state_dict": self.net.state_dict(), "meta": self.meta, "n_feat": self.content.shape[1]}, path)

    @classmethod
    def load(cls, path, snap: CatalogSnapshot) -> TwoTowerScorer:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=True)
        content = item_content_matrix(snap)
        if content.shape[1] != blob["n_feat"]:
            raise ValueError("two-tower feature layout does not match the active catalogue")
        net = _net(blob["n_feat"], blob["meta"]["embedding_dim"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return cls(net, content, blob["meta"])


def fit_two_tower(snap: CatalogSnapshot, train_cases, val_cases, negatives: int = 20, embedding_dim: int = 32,
                  epochs: int = 15, lr: float = 0.003, seed: int = 0, log=None) -> TwoTowerScorer:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    content = item_content_matrix(snap)
    net = _net(content.shape[1], embedding_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xc = torch.as_tensor(content)
    tr = [c for c in train_cases if c.target_eligible]
    va = [c for c in val_cases if c.target_eligible]
    history, best_state, best = [], None, -np.inf
    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(tr)
        tot = 0.0
        for start in range(0, len(tr), 64):
            batch = tr[start:start + 64]
            coll = torch.stack([Xc[c.visible].mean(0) for c in batch])
            cand = []
            for c in batch:
                pool = c.eligible[c.eligible != c.target]
                neg = rng.choice(pool, size=min(negatives, len(pool)), replace=False)
                cand.append(np.concatenate([[c.target], neg]))
            L = min(len(x) for x in cand)
            cand_idx = torch.as_tensor(np.stack([x[:L] for x in cand]))
            ce = net.encode_collection(coll)                       # (B, d)
            ie = net.encode_items(Xc[cand_idx.reshape(-1)]).reshape(len(batch), L, -1)
            logits = TwoTowerScorer.temperature * torch.einsum("bd,bld->bl", ce, ie)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(batch)
        net.eval()
        scorer = TwoTowerScorer(net, content, {})
        hits = []
        for c in va:
            s, _ = scorer.score(snap, c.profile, c.eligible)
            rank = int((s > s[np.flatnonzero(c.eligible == c.target)[0]]).sum())
            hits.append(rank < 8)
        v = float(np.mean(hits)) if hits else float("nan")
        history.append({"epoch": epoch, "train_loss": round(tot / max(1, len(tr)), 4), "val_hits@8": round(v, 4)})
        if log:
            log(f"[two-tower] epoch {epoch} loss={tot / max(1, len(tr)):.4f} val_hits@8={v:.4f}")
        if v > best:
            best = v
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    meta = {"scorer_id": SCORER_ID, "embedding_dim": embedding_dim, "negatives": negatives, "epochs": epochs,
            "best_val_hits@8": round(float(best), 4), "history": history, "train_cases": len(tr),
            "features": "multi-hot tags + episode bucket + score bucket", "loss": "sampled softmax"}
    return TwoTowerScorer(net, content, meta)
