# -*- coding: utf-8 -*-
"""
engine_svd.py — Candidate B: Funk-SVD matrix factorisation.

Slide spec:
  "r = mu + b_u + b_i + p_u . q_i / Users and titles share a 32-dim space /
   SGD fitting, no new dependencies / Centering becomes explicit as b_u and b_i"

Fit by stochastic gradient descent on the observed ratings (MSE + L2).
Explicit b_u / b_i make the user-mean centring an explicit, learned parameter.
"""
import time

import numpy as np
import torch
import torch.nn as nn

import config as C


class _FunkSVD(nn.Module):
    def __init__(self, n_users, n_items, dim, mu):
        super().__init__()
        self.mu = mu
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.pu = nn.Embedding(n_users, dim)
        self.qi = nn.Embedding(n_items, dim)
        for emb in (self.bu, self.bi, self.pu, self.qi):
            nn.init.normal_(emb.weight, 0.0, 0.05)

    def forward(self, u, i):
        return (self.bu(u).squeeze(1) + self.bi(i).squeeze(1)
                + (self.pu(u) * self.qi(i)).sum(1) + self.mu)


class SVDEngine:
    name = "SVD"

    def __init__(self, device=C.DEVICE):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.n_users = self.n_items = 0
        self.train_seconds = 0.0
        self.history = []

    def fit(self, R):
        t0 = time.time()
        R = R.tocsr()
        self.n_users, self.n_items = R.shape
        coo = R.tocoo()
        u = torch.from_numpy(coo.row.astype(np.int64))
        i = torch.from_numpy(coo.col.astype(np.int64))
        r = torch.from_numpy(coo.data.astype(np.float32))
        mu = float(r.mean())
        n = r.numel()

        # Reproducibility (same fix as NCF): model init (nn.init.normal_)
        # draws from the global stream — seed it right before construction;
        # SGD sample permutation uses a dedicated CPU generator below.
        torch.manual_seed(C.RANDOM_SEED)
        self.model = _FunkSVD(self.n_users, self.n_items, C.SVD_DIM, mu).to(self.device)
        opt = torch.optim.SGD(self.model.parameters(), lr=C.SVD_LR, momentum=0.9)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, C.SVD_EPOCHS // 3), gamma=0.5)
        lossf = nn.MSELoss()

        u_d = u.to(self.device); i_d = i.to(self.device); r_d = r.to(self.device)
        bs = C.SVD_BATCH
        nb = max(1, n // bs)
        gs = torch.Generator(device="cpu").manual_seed(C.RANDOM_SEED + 2)
        for ep in range(C.SVD_EPOCHS):
            ep_loss = 0.0
            perm = torch.randint(0, n, (nb * bs,), generator=gs).to(self.device)
            for b in range(nb):
                sl = perm[b * bs:(b + 1) * bs]
                ub, ib, rb = u_d[sl], i_d[sl], r_d[sl]
                pred = self.model(ub, ib)
                loss = lossf(pred, rb)
                # L2 on the latent factors only (standard Funk-SVD regularisation)
                loss = loss + C.SVD_L2 * (self.model.pu(ub).pow(2).sum(1).mean()
                                          + self.model.qi(ib).pow(2).sum(1).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
            sched.step()
            rmse = float(np.sqrt(ep_loss / nb))
            self.history.append(dict(epoch=ep + 1, train_rmse=rmse, lr=sched.get_last_lr()[0]))
            print(f"[SVD] epoch {ep+1}/{C.SVD_EPOCHS} train_RMSE={rmse:.4f}", flush=True)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def predict_pairs(self, u_idx, i_idx, chunk=200000, clip=True):
        m = self.model
        out = np.empty(len(u_idx), dtype=np.float32)
        for s in range(0, len(u_idx), chunk):
            e = min(s + chunk, len(u_idx))
            ub = torch.from_numpy(np.asarray(u_idx[s:e], dtype=np.int64)).to(self.device)
            ib = torch.from_numpy(np.asarray(i_idx[s:e], dtype=np.int64)).to(self.device)
            out[s:e] = m(ub, ib).float().cpu().numpy()
        if clip:
            np.clip(out, C.RATING_MIN, C.RATING_MAX, out=out)
        return out

    @torch.no_grad()
    def topk(self, u_idx, k, R_train=None, chunk=2000):
        m = self.model
        uu = np.asarray(u_idx, dtype=np.int64)
        n = len(uu)
        out_i = np.zeros((n, k), dtype=np.int32)
        out_s = np.zeros((n, k), dtype=np.float32)
        bi = m.bi.weight.squeeze(1)
        qi = m.qi.weight
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            ub = torch.from_numpy(uu[s:e]).to(self.device)
            sc = (m.bu(ub) + m.bi.weight.squeeze(1).unsqueeze(0)).float()          # (c, I)
            sc = sc + m.pu(ub) @ qi.t() + m.mu
            sc = sc.cpu().numpy()
            if R_train is not None:
                for r_i, u in enumerate(uu[s:e]):
                    a, b = R_train.indptr[u], R_train.indptr[u + 1]
                    sc[r_i, R_train.indices[a:b]] = -np.inf
            part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sc, part, axis=1)
            order = np.argsort(-vals, axis=1)
            part = np.take_along_axis(part, order, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
            out_i[s:e] = part
            out_s[s:e] = vals
        return out_i, out_s
