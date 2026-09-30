"""Algorithm 3 recall C: Neural Collaborative Filtering, explicit-rating variant (ncf_explicit).

GMF branch (elementwise product of user/item embeddings) and MLP branch
(concatenated embeddings -> MLP) are fused by a linear layer into one logit;
the output is a 1-10 rating: r_hat = 1 + 9 * sigmoid(logit). Trained with a
Huber (or MSE) loss, dropout, weight decay and validation early stopping, so it
is comparable with ICF / MF on MAE and RMSE.

The model is ID-only: an application user who is not in the training data is
reported as ``unsupported_new_user`` and the caller falls back to ICF/content.
No random user vector is ever invented. Requires the optional ``ml`` extra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

ENGINE_ID = "ncf_explicit"
ENGINE_VERSION = "1.0"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _build(n_users: int, n_items: int, dim: int, layers: list[int], dropout: float):
    import torch
    from torch import nn

    class NCF(nn.Module):
        def __init__(self):
            super().__init__()
            self.ug = nn.Embedding(n_users, dim)
            self.ig = nn.Embedding(n_items, dim)
            self.um = nn.Embedding(n_users, dim)
            self.im = nn.Embedding(n_items, dim)
            mlp, inp = [], 2 * dim
            for h in layers:
                mlp += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
                inp = h
            self.mlp = nn.Sequential(*mlp)
            self.out = nn.Linear(dim + inp, 1)
            for emb in (self.ug, self.ig, self.um, self.im):
                nn.init.normal_(emb.weight, std=0.05)

        def forward(self, u, i):
            g = self.ug(u) * self.ig(i)
            m = self.mlp(torch.cat([self.um(u), self.im(i)], dim=1))
            logit = self.out(torch.cat([g, m], dim=1)).squeeze(1)
            return 1.0 + 9.0 * torch.sigmoid(logit)

    return NCF()


@dataclass
class NCFModel:
    net: object
    n_users: int
    n_items: int
    params: dict
    history: list

    def predict_pairs(self, u: np.ndarray, i: np.ndarray, batch: int = 262144) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(u), batch):
                uu = torch.as_tensor(np.asarray(u[s:s + batch], dtype=np.int64))
                ii = torch.as_tensor(np.asarray(i[s:s + batch], dtype=np.int64))
                out.append(self.net(uu, ii).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_user(self, u: int) -> np.ndarray:
        items = np.arange(self.n_items)
        return self.predict_pairs(np.full(self.n_items, u), items)

    def save(self, path) -> None:
        import torch

        torch.save({"state_dict": self.net.state_dict(), "n_users": self.n_users, "n_items": self.n_items,
                    "params": self.params, "history": self.history}, path)

    @classmethod
    def load(cls, path) -> NCFModel:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=True)
        p = blob["params"]
        net = _build(blob["n_users"], blob["n_items"], p["embedding_dim"], p["mlp_layers"], p["dropout"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return cls(net, blob["n_users"], blob["n_items"], p, blob["history"])


def fit_ncf(u, i, r, n_users: int, n_items: int, val=None, embedding_dim: int = 32,
            mlp_layers: list[int] | None = None, dropout: float = 0.2, weight_decay: float = 1e-5,
            learning_rate: float = 0.002, batch_size: int = 8192, epochs: int = 20, patience: int = 3,
            loss: str = "huber", seed: int = 0, log=None, max_train_seconds: float | None = None, **_) -> NCFModel:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    layers = list(mlp_layers or [64, 32, 16])
    net = _build(n_users, n_items, embedding_dim, layers, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    crit = nn.HuberLoss(delta=1.0) if loss == "huber" else nn.MSELoss()
    U = torch.as_tensor(np.asarray(u, dtype=np.int64))
    I = torch.as_tensor(np.asarray(i, dtype=np.int64))
    R = torch.as_tensor(np.asarray(r, dtype=np.float32))
    n = len(R)
    best_state, best_rmse, bad = None, np.inf, 0
    history = []
    t0 = time.time()
    params = {"embedding_dim": embedding_dim, "mlp_layers": layers, "dropout": dropout,
              "weight_decay": weight_decay, "learning_rate": learning_rate, "batch_size": batch_size,
              "loss": loss, "seed": seed, "output": "1 + 9*sigmoid(logit)"}
    model = NCFModel(net, n_users, n_items, params, history)
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.as_tensor(rng.permutation(n))
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            pred = net(U[idx], I[idx])
            lossv = crit(pred, R[idx])
            opt.zero_grad()
            lossv.backward()
            opt.step()
            tot += lossv.item() * len(idx)
        va = float("nan")
        if val is not None and len(val[2]):
            p = model.predict_pairs(val[0], val[1])
            va = float(np.sqrt(np.mean((p - np.asarray(val[2], dtype=np.float64)) ** 2)))
        history.append({"epoch": epoch, "train_loss": round(tot / n, 4), "val_rmse": round(va, 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        if log:
            log(f"[ncf] epoch {epoch} train_loss={tot / n:.4f} val_rmse={va:.4f} ({time.time() - t0:.0f}s)")
        if val is None:
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif va < best_rmse - 1e-4:
            best_rmse, bad = va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if max_train_seconds and time.time() - t0 > max_train_seconds:
            history[-1]["stopped"] = "time_budget"
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    params.update({"epochs_run": len(history),
                   "best_val_rmse": None if not np.isfinite(best_rmse) else round(float(best_rmse), 4)})
    return NCFModel(net, n_users, n_items, params, history)
