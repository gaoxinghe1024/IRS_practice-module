"""Algorithm 3 recall B: biased matrix factorisation (explicit ratings).

    r_hat(u,i) = mu + b_u + b_i + p_u . q_i
    loss = sum_{observed train (u,i)} (r_ui - r_hat)^2 + reg * (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)

Fitted with shuffled mini-batch SGD using lazy (row-wise) Adam updates in numpy,
early-stopped on a validation RMSE. mu, biases and factors all come from train.

New application users are folded in by ridge regression with the item side
fixed:  min_w sum_i (r_ui - mu - b_i - [q_i, 1] . w)^2 + fold_in_reg * |w|^2,
w = [p_u, b_u]. With fewer than ``fold_in_min_ratings`` known items the model
returns None and the caller falls back to ICF / content.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

ENGINE_ID = "mf_biased_sgd"
ENGINE_VERSION = "1.0"


@dataclass
class MFModel:
    mu: float
    bu: np.ndarray
    bi: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    params: dict
    history: list

    def predict_pairs(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])

    def predict_user(self, u: int) -> np.ndarray:
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]

    def fold_in(self, items: np.ndarray, ratings: np.ndarray, reg: float = 1.0, min_ratings: int = 3
                ) -> tuple[np.ndarray, float] | None:
        items = np.asarray(items, dtype=np.int64)
        if len(items) < min_ratings:
            return None
        X = np.hstack([self.Q[items], np.ones((len(items), 1))])
        y = np.asarray(ratings, dtype=np.float64) - self.mu - self.bi[items]
        A = X.T @ X + reg * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        return w[:-1], float(w[-1])

    def predict_folded(self, p: np.ndarray, b_u: float) -> np.ndarray:
        return self.mu + b_u + self.bi + self.Q @ p

    def save(self, path) -> None:
        np.savez_compressed(path, mu=self.mu, bu=self.bu, bi=self.bi, P=self.P, Q=self.Q,
                            params=np.array([repr(self.params)]), history=np.array([repr(self.history)]))

    @classmethod
    def load(cls, path) -> MFModel:
        import ast

        z = np.load(path, allow_pickle=False)
        return cls(mu=float(z["mu"]), bu=z["bu"], bi=z["bi"], P=z["P"], Q=z["Q"],
                   params=ast.literal_eval(str(z["params"][0])), history=ast.literal_eval(str(z["history"][0])))


class _LazyAdam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0

    def step(self, param: np.ndarray, rows: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1
        m = self.b1 * self.m[rows] + (1 - self.b1) * grad
        v = self.b2 * self.v[rows] + (1 - self.b2) * grad * grad
        self.m[rows], self.v[rows] = m, v
        mhat = m / (1 - self.b1 ** self.t)
        vhat = v / (1 - self.b2 ** self.t)
        param[rows] -= (self.lr * mhat / (np.sqrt(vhat) + self.eps)).astype(param.dtype)


def _aggregate(inv: np.ndarray, n_rows: int, values: np.ndarray) -> np.ndarray:
    agg = sp.csr_matrix((np.ones(len(inv), dtype=np.float32), (inv, np.arange(len(inv)))), shape=(n_rows, len(inv)))
    return np.asarray(agg @ values)


def rmse(model: MFModel, u, i, r) -> float:
    if len(r) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.clip(model.predict_pairs(u, i), 1, 10) - r) ** 2)))


def fit_mf(u: np.ndarray, i: np.ndarray, r: np.ndarray, n_users: int, n_items: int,
           val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, latent_dim: int = 32, epochs: int = 30,
           batch_size: int = 65536, learning_rate: float = 0.01, reg: float = 0.04, patience: int = 3,
           seed: int = 0, log=None, **_) -> MFModel:
    rng = np.random.default_rng(seed)
    u = np.asarray(u, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)
    r = np.asarray(r, dtype=np.float32)
    mu = float(r.mean())
    P = rng.normal(0, 0.05, (n_users, latent_dim)).astype(np.float32)
    Q = rng.normal(0, 0.05, (n_items, latent_dim)).astype(np.float32)
    bu = np.zeros((n_users, 1), dtype=np.float32)
    bi = np.zeros((n_items, 1), dtype=np.float32)
    opt = {k: _LazyAdam(s.shape, learning_rate) for k, s in (("P", P), ("Q", Q), ("bu", bu), ("bi", bi))}
    best = None
    best_rmse = np.inf
    bad = 0
    history = []
    n = len(r)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        sq = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            ub, ib, rb = u[idx], i[idx], r[idx]
            pu, qi = P[ub], Q[ib]
            pred = mu + bu[ub, 0] + bi[ib, 0] + np.einsum("ij,ij->i", pu, qi)
            e = rb - pred
            sq += float(e @ e)
            scale = 1.0 / len(idx)
            gP = (-e[:, None] * qi + reg * pu) * scale
            gQ = (-e[:, None] * pu + reg * qi) * scale
            gbu = ((-e + reg * bu[ub, 0]) * scale)[:, None]
            gbi = ((-e + reg * bi[ib, 0]) * scale)[:, None]
            uu, uinv = np.unique(ub, return_inverse=True)
            ii, iinv = np.unique(ib, return_inverse=True)
            opt["P"].step(P, uu, _aggregate(uinv, len(uu), gP))
            opt["bu"].step(bu, uu, _aggregate(uinv, len(uu), gbu))
            opt["Q"].step(Q, ii, _aggregate(iinv, len(ii), gQ))
            opt["bi"].step(bi, ii, _aggregate(iinv, len(ii), gbi))
        model = MFModel(mu, bu[:, 0].copy(), bi[:, 0].copy(), P.copy(), Q.copy(), {}, [])
        tr_rmse = float(np.sqrt(sq / n))
        va = rmse(model, *val) if val is not None else float("nan")
        history.append({"epoch": epoch, "train_rmse": round(tr_rmse, 4), "val_rmse": round(va, 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        if log:
            log(f"[mf] epoch {epoch} train_rmse={tr_rmse:.4f} val_rmse={va:.4f} ({time.time() - t0:.0f}s)")
        if val is None:
            best = model
            continue
        if va < best_rmse - 1e-4:
            best_rmse, best, bad = va, model, 0
        else:
            bad += 1
            if bad >= patience:
                break
    assert best is not None
    best.params = {"latent_dim": latent_dim, "epochs_run": len(history), "batch_size": batch_size,
                   "learning_rate": learning_rate, "reg": reg, "seed": seed, "optimizer": "minibatch lazy Adam",
                   "best_val_rmse": None if not np.isfinite(best_rmse) else round(float(best_rmse), 4)}
    best.history = history
    return best
