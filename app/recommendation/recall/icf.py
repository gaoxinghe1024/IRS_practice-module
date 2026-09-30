"""Algorithm 3 recall A: item-based collaborative filtering (exact adjusted cosine).

Training (offline, train split only):
    mu_u  = mean of user u's observed training ratings
    a_ui  = r_ui - mu_u (observed entries only; a separate observed mask M keeps
            observed entries whose centred value is exactly 0)
    s_raw(i,j) = sum_{u in U_ij} a_ui a_uj / sqrt(sum_{U_ij} a_ui^2 * sum_{U_ij} a_uj^2)
                 U_ij = users who rated both i and j (co-rated denominator, not full column norms)
    kept when c_ij = |U_ij| >= min_co_rated; s = s_raw * c/(c + shrinkage); top-N positive per item.

Computed per item block with sparse products:
    numerator N = A_b^T A,  co-count C = M_b^T M,  P = (A o A)^T M  (P[i,j] = sum_{U_ij} a_ui^2)
    denominator(i,j) = sqrt(P[i,j] * P[j,i])

Prediction (online or offline):
    r_hat(u,i) = mu_u + sum_{j in R_u & N(i)} s(i,j)(r_uj - mu_u) / (sum |s(i,j)| + damping)
    clipped to [1, 10]; items without neighbour support get no ICF prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

ENGINE_ID = "icf_adjusted_cosine"
ENGINE_VERSION = "1.0"


@dataclass
class ICFModel:
    S: sp.csr_matrix          # items x items, row i = neighbours N(i) with s(i, j) > 0
    S_T: sp.csr_matrix        # transpose (row j = items i having j as a neighbour)
    co_counts: sp.csr_matrix  # same sparsity as S, number of co-rating users
    n_items: int
    params: dict

    @property
    def nnz(self) -> int:
        return int(self.S.nnz)

    def save(self, path) -> None:
        np.savez_compressed(path, S_data=self.S.data, S_indices=self.S.indices, S_indptr=self.S.indptr,
                            C_data=self.co_counts.data, n_items=self.n_items,
                            params=np.array([repr(self.params)]))

    @classmethod
    def load(cls, path) -> ICFModel:
        z = np.load(path, allow_pickle=False)
        n = int(z["n_items"])
        S = sp.csr_matrix((z["S_data"], z["S_indices"], z["S_indptr"]), shape=(n, n))
        C = sp.csr_matrix((z["C_data"], z["S_indices"], z["S_indptr"]), shape=(n, n))
        import ast

        params = ast.literal_eval(str(z["params"][0]))
        return cls(S=S, S_T=S.T.tocsr(), co_counts=C, n_items=n, params=params)


def build_matrices(users: np.ndarray, items: np.ndarray, ratings: np.ndarray, n_users: int, n_items: int
                   ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Centred rating matrix A (explicit zeros kept), observed mask M (int32), user means."""
    ratings = ratings.astype(np.float64)
    cnt = np.bincount(users, minlength=n_users)
    sums = np.bincount(users, weights=ratings, minlength=n_users)
    mu = np.divide(sums, cnt, out=np.zeros(n_users), where=cnt > 0)
    centred = (ratings - mu[users]).astype(np.float32)
    A = sp.csr_matrix((centred, (users, items)), shape=(n_users, n_items))
    # csr_matrix((data,(r,c))) sums duplicates and keeps explicit zeros; build M the same way
    M = sp.csr_matrix((np.ones(len(users), dtype=np.int32), (users, items)), shape=(n_users, n_items))
    return A, M, mu


def fit_icf(users: np.ndarray, items: np.ndarray, ratings: np.ndarray, n_users: int, n_items: int,
            min_co_rated: int = 3, shrinkage: float = 10.0, top_n: int = 150, block_size: int = 1024,
            log=None) -> ICFModel:
    t0 = time.time()
    A, M, _mu = build_matrices(users, items, ratings, n_users, n_items)
    A2 = A.multiply(A).tocsr().astype(np.float32)
    A_csc, M_csc, A2_csc = A.tocsc(), M.tocsc(), A2.tocsc()
    rows_out, cols_out, vals_out, cnt_out = [], [], [], []
    for start in range(0, n_items, block_size):
        stop = min(n_items, start + block_size)
        Ab = A_csc[:, start:stop]
        Mb = M_csc[:, start:stop]
        A2b = A2_csc[:, start:stop]
        N = (Ab.T @ A).toarray()                      # (b, n_items) float32
        C = (Mb.T @ M).toarray().astype(np.int32)     # co-rating counts
        P1 = (A2b.T @ M).toarray()                    # sum_{U_ij} a_ui^2
        P2 = (Mb.T @ A2).toarray()                    # sum_{U_ij} a_uj^2
        den = np.sqrt(P1.astype(np.float64) * P2.astype(np.float64))
        valid = (C >= min_co_rated) & (den > 0)
        S = np.zeros_like(N, dtype=np.float64)
        np.divide(N, den, out=S, where=valid)
        S *= C / (C + shrinkage) if shrinkage > 0 else 1.0
        S[~valid] = 0.0
        S[np.arange(stop - start), np.arange(start, stop)] = 0.0   # no self-similarity
        S = np.clip(S, -1.0, 1.0)
        for r in range(stop - start):
            row = S[r]
            pos = np.flatnonzero(row > 0)
            if pos.size == 0:
                continue
            if pos.size > top_n:
                part = np.argpartition(-row[pos], top_n - 1)[:top_n]
                pos = pos[part]
            order = np.lexsort((pos, -row[pos]))
            pos = pos[order]
            rows_out.append(np.full(pos.size, start + r, dtype=np.int32))
            cols_out.append(pos.astype(np.int32))
            vals_out.append(row[pos].astype(np.float32))
            cnt_out.append(C[r, pos].astype(np.float32))
        if log and ((start // block_size) % 4 == 0):
            log(f"[icf] items {stop}/{n_items} elapsed={time.time() - t0:.1f}s")
    if rows_out:
        r = np.concatenate(rows_out)
        c = np.concatenate(cols_out)
        v = np.concatenate(vals_out)
        k = np.concatenate(cnt_out)
    else:
        r = c = np.zeros(0, dtype=np.int32)
        v = k = np.zeros(0, dtype=np.float32)
    S = sp.csr_matrix((v, (r, c)), shape=(n_items, n_items))
    Cm = sp.csr_matrix((k, (r, c)), shape=(n_items, n_items))
    # align co-count sparsity with S (same construction order -> same structure)
    S.sort_indices()
    Cm.sort_indices()
    params = {"min_co_rated": min_co_rated, "shrinkage": shrinkage, "top_n": top_n,
              "definition": "exact adjusted cosine over co-rated users"}
    if log:
        log(f"[icf] done: nnz={S.nnz} in {time.time() - t0:.1f}s")
    return ICFModel(S=S, S_T=S.T.tocsr(), co_counts=Cm, n_items=n_items, params=params)


def exact_adjusted_cosine_dense(R: np.ndarray, observed: np.ndarray, min_co_rated: int = 3,
                                shrinkage: float = 0.0) -> np.ndarray:
    """Brute-force oracle on a small dense matrix (tests only)."""
    n_users, n_items = R.shape
    mu = np.array([R[u, observed[u]].mean() if observed[u].any() else 0.0 for u in range(n_users)])
    A = np.where(observed, R - mu[:, None], 0.0)
    S = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            U = observed[:, i] & observed[:, j]
            c = int(U.sum())
            if c < min_co_rated:
                continue
            num = float((A[U, i] * A[U, j]).sum())
            den = float(np.sqrt((A[U, i] ** 2).sum() * (A[U, j] ** 2).sum()))
            if den <= 0:
                continue
            s = num / den
            if shrinkage > 0:
                s *= c / (c + shrinkage)
            S[i, j] = s
    return S


@dataclass
class ICFPrediction:
    items: np.ndarray        # model item indices with support
    predicted: np.ndarray    # clipped to [1, 10]
    support: np.ndarray      # sum |s| over supporting rated neighbours
    n_support: np.ndarray    # number of supporting rated neighbours


class ICFScorer:
    engine_id = ENGINE_ID
    version = ENGINE_VERSION

    def __init__(self, model: ICFModel, damping: float = 0.0):
        self.model = model
        self.damping = float(damping)

    def predict(self, rated_items: np.ndarray, rated_values: np.ndarray, user_mean: float | None = None
                ) -> ICFPrediction:
        rated_items = np.asarray(rated_items, dtype=np.int64)
        rated_values = np.asarray(rated_values, dtype=np.float64)
        if rated_items.size == 0:
            z = np.zeros(0)
            return ICFPrediction(np.zeros(0, dtype=np.int64), z, z, z)
        mu = float(rated_values.mean()) if user_mean is None else float(user_mean)
        centred = rated_values - mu
        sub = self.model.S_T[rated_items]              # rows j in R_u; columns = target items i
        num = np.asarray(sub.T @ centred).ravel()
        absw = sub.copy()
        absw.data = np.abs(absw.data)
        den = np.asarray(absw.T @ np.ones(len(rated_items))).ravel()
        cnt_m = sub.copy()
        cnt_m.data = np.ones_like(cnt_m.data)
        nsup = np.asarray(cnt_m.T @ np.ones(len(rated_items))).ravel()
        has = den > 0
        items = np.flatnonzero(has)
        pred = mu + num[has] / (den[has] + self.damping)
        return ICFPrediction(items, np.clip(pred, 1.0, 10.0), den[has], nsup[has])

    def explain(self, item: int, rated_items: np.ndarray, rated_values: np.ndarray, user_mean: float,
                top: int = 3) -> list[dict]:
        """Rated neighbours of `item` ranked by |s(i,j) * (r_uj - mu_u)|."""
        row = self.model.S[item]
        crow = self.model.co_counts[item]
        nb = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
        cc = dict(zip(crow.indices.tolist(), crow.data.tolist(), strict=True))
        out = []
        for j, r in zip(np.asarray(rated_items).tolist(), np.asarray(rated_values).tolist(), strict=True):
            if j in nb:
                out.append({"item": j, "similarity": round(nb[j], 4), "co_rated_users": int(cc.get(j, 0)),
                            "your_rating": r, "contribution": round(nb[j] * (r - user_mean), 4)})
        out.sort(key=lambda d: -abs(d["contribution"]))
        return out[:top]
