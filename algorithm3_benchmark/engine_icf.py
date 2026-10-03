# -*- coding: utf-8 -*-
"""
engine_icf.py — Candidate A: item-based collaborative filtering (ICF).

Slide spec:
  "Center ratings by subtracting each user's mean / Build item-item co-occurrence,
   compute cosine / Seed with >= 8 rated titles, merge neighbours, recall Top-20"

Implementation notes
  * The item-item cosine Gram matrix is built block-wise over items and pruned to
    the top-ICF_NEIGHBOURS neighbours per item. Blocking keeps peak memory small
    enough for a 16 GB machine while 14,202 x 14,202 would be ~800 MB dense.
  * Neighbours are kept by |cosine| so strongly *negative* item relations also
    count, which is what a centred (mean-subtracted) matrix is meant to expose.
  * Rating prediction uses the standard item-based weighted average.
"""
import time

import numpy as np
import scipy.sparse as sp

import config as C


class ICFEngine:
    name = "ICF"

    def __init__(self):
        self.user_mean = None
        self.Rc = None          # centred matrix (CSR)
        self.R = None           # raw training matrix (CSR)
        self.S = None           # pruned item-item cosine (CSR)
        self.train_seconds = 0.0
        self.sim_nnz = 0

    # ---------------------------------------------------------------- fit
    def fit(self, R):
        t0 = time.time()
        R = R.tocsr()
        self.R = R
        cnt = np.diff(R.indptr).astype(np.float64)
        self.user_mean = (np.asarray(R.sum(1)).ravel() / np.maximum(cnt, 1)).astype(np.float32)

        # 1) centre each user's ratings by that user's mean (observed entries only)
        Rc = R.copy()
        Rc.data = Rc.data - np.repeat(self.user_mean, cnt.astype(np.int64))
        self.Rc = Rc.tocsr()
        self.Rc.sort_indices()

        # 2) column-normalise -> cosine becomes a plain Gram product
        colnorm = np.sqrt(np.asarray(self.Rc.multiply(self.Rc).sum(0)).ravel()).astype(np.float32)
        colnorm[colnorm == 0] = 1.0
        A = (self.Rc @ sp.diags(1.0 / colnorm)).tocsr()
        A.sort_indices()

        # 3) block-wise item-item cosine, pruned to top-K neighbours
        n_items = A.shape[1]
        k = C.ICF_NEIGHBOURS
        rows_out, cols_out, vals_out = [], [], []
        for lo in range(0, n_items, C.ICF_BLOCK):
            hi = min(lo + C.ICF_BLOCK, n_items)
            B = (A[:, lo:hi].T.tocsr() @ A).toarray()      # (bs, n_items) dense
            for r in range(hi - lo):
                B[r, lo + r] = 0.0                          # drop self-similarity
            kk = min(k, n_items - 1)
            part = np.argpartition(np.abs(B), -kk, axis=1)[:, -kk:]
            rr = np.repeat(np.arange(lo, hi), kk)
            cc = part.ravel()
            vv = B[np.repeat(np.arange(hi - lo), kk), cc]
            rows_out.append(rr)
            cols_out.append(cc)
            vals_out.append(vv)
        S = sp.csr_matrix((np.concatenate(vals_out).astype(np.float32),
                           (np.concatenate(rows_out), np.concatenate(cols_out))),
                          shape=(n_items, n_items))
        S.sort_indices()
        self.S = S
        self.sim_nnz = S.nnz
        self.train_seconds = time.time() - t0
        return self

    # ---------------------------------------------------------------- predict
    def predict_pairs(self, u_idx, i_idx, chunk=20000, clip=True):
        """Weighted-average item-based prediction for (user, item) pairs."""
        out = np.empty(len(u_idx), dtype=np.float32)
        for s in range(0, len(u_idx), chunk):
            e = min(s + chunk, len(u_idx))
            uu = np.asarray(u_idx[s:e], dtype=np.int64)
            ii = np.asarray(i_idx[s:e], dtype=np.int64)
            sel = self.S[ii]                       # (n, n_items) neighbour rows
            prof = self.Rc[uu]                     # (n, n_items) centred profile
            num = np.asarray(sel.multiply(prof).sum(axis=1)).ravel()
            den = np.asarray(np.abs(sel).sum(axis=1)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                adj = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
            out[s:e] = self.user_mean[uu] + adj
        if clip:
            np.clip(out, C.RATING_MIN, C.RATING_MAX, out=out)
        return out

    def score_items(self, u_idx, chunk=1000):
        """Full-catalogue scores for a chunk of users -> dense (n, n_items)."""
        uu = np.asarray(u_idx, dtype=np.int64)
        P = self.Rc[uu]
        sc = (P @ self.S).toarray().astype(np.float32)
        return sc

    # ---------------------------------------------------------------- top-N
    def topk(self, u_idx, k, R_train=None, chunk=1000):
        uu = np.asarray(u_idx, dtype=np.int64)
        R = R_train if R_train is not None else self.R
        n = len(uu)
        out_i = np.zeros((n, k), dtype=np.int32)
        out_s = np.zeros((n, k), dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sc = self.score_items(uu[s:e])
            # never recommend something the user already rated in training
            for r, u in enumerate(uu[s:e]):
                a, b = R.indptr[u], R.indptr[u + 1]
                sc[r, R.indices[a:b]] = -np.inf
            part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sc, part, axis=1)
            order = np.argsort(-vals, axis=1)
            part = np.take_along_axis(part, order, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
            out_i[s:e] = part
            out_s[s:e] = vals + self.user_mean[uu[s:e], None]
        return out_i, out_s
