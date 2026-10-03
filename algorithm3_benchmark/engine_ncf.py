# -*- coding: utf-8 -*-
"""
engine_ncf.py — Candidate C: NeuralCF (neural-network collaborative filtering).

Slide spec:
  "User/title embeddings concatenated into an MLP / Network outputs the score /
   a GMF branch is a learnable dot product / Two branches fused and normalized,
   BCE loss + early stop"
  Slide also records the expected outcome: "Must overfit on sparse data - kept as
   a negative example", so this engine is deliberately kept as the negative control.

NCF is an implicit-feedback ranker, so it emits a preference probability, not a
rating. For the MAE column a linear calibration (least squares fitted on the
training split only) maps that probability back onto the 1-10 rating scale; this
is documented in the test report as an implementation choice.
"""
import time

import numpy as np
import torch
import torch.nn as nn

import config as C


class _NeuMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.gmf_u = nn.Embedding(n_users, dim)
        self.gmf_i = nn.Embedding(n_items, dim)
        self.mlp_u = nn.Embedding(n_users, dim)
        self.mlp_i = nn.Embedding(n_items, dim)
        for e in (self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i):
            nn.init.normal_(e.weight, 0.0, 0.05)
        # GMF branch: learnable (weighted) dot product
        self.gmf_out = nn.Linear(dim, 1)
        # MLP branch: concatenated embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.ReLU(),
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, 1),
        )
        # fuse the two branches and squash to a probability
        self.fuse = nn.Linear(2, 1)

    def forward(self, u, i):
        g = self.gmf_out(self.gmf_u(u) * self.gmf_i(i))
        m = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=1))
        return self.fuse(torch.cat([g, m], dim=1)).squeeze(1)


class NCFEngine:
    name = "NCF"

    def __init__(self, device=C.DEVICE):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.calib_a = 1.0
        self.calib_b = 5.5
        self.train_seconds = 0.0
        self.history = []

    # ---------------------------------------------------------------- fit
    def fit(self, R, val_df=None, users_map=None):
        t0 = time.time()
        R = R.tocsr()
        self.n_users, self.n_items = R.shape
        coo = R.tocoo()
        u_all = coo.row.astype(np.int64)
        i_all = coo.col.astype(np.int64)
        r_all = coo.data.astype(np.float32)
        pos_mask = r_all >= C.NCF_POS_THRESHOLD
        pos_u = torch.from_numpy(u_all[pos_mask])
        pos_i = torch.from_numpy(i_all[pos_mask])
        n_pos = pos_u.numel()
        print(f"[NCF] positives (rating>={C.NCF_POS_THRESHOLD}): {n_pos:,}", flush=True)

        # Fix ALL stochastic sources for reproducibility:
        #   1) model init — nn.init.normal_ and nn.Linear's default init draw
        #      from the global stream, so seed it right before construction;
        #   2) negative sampling — dedicated CPU generator gneg (below);
        #   3) permutation / AUC / calibration — already seeded generators.
        torch.manual_seed(C.RANDOM_SEED)
        self.model = _NeuMF(self.n_users, self.n_items, C.NCF_EMB).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=C.NCF_LR)
        lossf = nn.BCEWithLogitsLoss()

        g = torch.Generator(device="cpu").manual_seed(C.RANDOM_SEED)
        # Negative sampling must be reproducible too: without a generator,
        # torch.randint draws from the global CUDA stream, so every retrain
        # produces a different model (observed: AUC ~equal but sampled-pool
        # HR differing 3x). A dedicated CPU generator makes retraining
        # bit-identical on the same machine, independent of device.
        gneg = torch.Generator(device="cpu").manual_seed(C.RANDOM_SEED + 1)
        cap = min(C.NCF_MAX_POS, n_pos)
        bs = C.NCF_BATCH
        best_auc, best_state, bad = -1.0, None, 0

        for ep in range(C.NCF_EPOCHS):
            perm = torch.randperm(n_pos, generator=g)[:cap]
            pu = pos_u[perm]; pi = pos_i[perm]
            nb = max(1, cap // bs)
            ep_loss = 0.0
            for b in range(nb):
                s, e = b * bs, (b + 1) * bs
                ub = pu[s:e].to(self.device); ib = pi[s:e].to(self.device)
                nb_neg = ub.numel() * C.NCF_NEG
                nu = torch.randint(0, self.n_users, (nb_neg,), generator=gneg).to(self.device)
                ni = torch.randint(0, self.n_items, (nb_neg,), generator=gneg).to(self.device)
                all_u = torch.cat([ub, nu]); all_i = torch.cat([ib, ni])
                y = torch.cat([torch.ones(ub.numel(), device=self.device),
                               torch.zeros(nb_neg, device=self.device)])
                logit = self.model(all_u, all_i)
                loss = lossf(logit, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
            auc = self._quick_auc(R)
            self.history.append(dict(epoch=ep + 1, bce=ep_loss / nb, holdout_auc=auc))
            print(f"[NCF] epoch {ep+1}/{C.NCF_EPOCHS} BCE={ep_loss/nb:.4f} "
                  f"holdout_AUC={auc:.4f}", flush=True)
            if auc > best_auc + 1e-4:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= C.NCF_PATIENCE:
                    print(f"[NCF] early stop at epoch {ep+1} (best AUC={best_auc:.4f})", flush=True)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._calibrate(u_all, i_all, r_all)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def _quick_auc(self, R, n_sample=200000):
        """Ranking AUC on a random sample of training positives vs random negatives."""
        g = np.random.default_rng(C.RANDOM_SEED)
        coo = R.tocoo()
        n = coo.data.size
        idx = g.choice(n, size=min(n_sample, n), replace=False)
        uu = coo.row[idx]; ii = coo.col[idx]
        pos = coo.data[idx] >= C.NCF_POS_THRESHOLD
        nu = g.integers(0, self.n_users, size=idx.size); ni = g.integers(0, self.n_items, size=idx.size)
        all_u = np.concatenate([uu, nu]); all_i = np.concatenate([ii, ni])
        y = np.concatenate([pos.astype(np.int8), np.zeros(idx.size, dtype=np.int8)])
        s = self.predict_pairs(all_u, all_i, chunk=200000, clip=False)
        P, N = s[y == 1], s[y == 0]
        if len(P) == 0 or len(N) == 0:
            return 0.5
        # Mann-Whitney U statistic, sub-sampled to bound cost
        m = min(len(P), len(N), 50000)
        P = P[:m]; N = N[:m]
        return float(((P[:, None] > N[None, :]).mean() + 0.5 * (P[:, None] == N[None, :]).mean()))

    @torch.no_grad()
    def _calibrate(self, u_all, i_all, r_all, n_sample=400000):
        """Least-squares linear map from NCF score -> 1-10 rating (training split only)."""
        g = np.random.default_rng(C.RANDOM_SEED)
        idx = g.choice(r_all.size, size=min(n_sample, r_all.size), replace=False)
        s = self.predict_pairs(u_all[idx], i_all[idx], chunk=200000, clip=False)
        a, b = np.polyfit(s, r_all[idx], 1)
        self.calib_a, self.calib_b = float(a), float(b)
        pred = a * s + b
        print(f"[NCF] rating calibration: r_hat = {a:.3f}*score + {b:.3f} "
              f"(train MAE={np.abs(np.clip(pred,1,10)-r_all[idx]).mean():.4f})", flush=True)

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def _score(self, u_idx, i_idx, chunk=400000):
        out = np.empty(len(u_idx), dtype=np.float32)
        for s in range(0, len(u_idx), chunk):
            e = min(s + chunk, len(u_idx))
            ub = torch.from_numpy(np.asarray(u_idx[s:e], dtype=np.int64)).to(self.device)
            ib = torch.from_numpy(np.asarray(i_idx[s:e], dtype=np.int64)).to(self.device)
            out[s:e] = self.model(ub, ib).float().cpu().numpy()
        return out

    def predict_pairs(self, u_idx, i_idx, chunk=400000, clip=True):
        p = self._score(u_idx, i_idx, chunk)
        r = self.calib_a * p + self.calib_b
        if clip:
            np.clip(r, C.RATING_MIN, C.RATING_MAX, out=r)
        r = r.astype(np.float32)
        return r

    def raw_scores(self, u_idx, i_idx, chunk=400000):
        return self._score(u_idx, i_idx, chunk)

    # ---------------------------------------------------------------- top-N
    @torch.no_grad()
    def topk(self, u_idx, k, R_train=None, chunk=500):
        uu = np.asarray(u_idx, dtype=np.int64)
        n = len(uu)
        out_i = np.zeros((n, k), dtype=np.int32)
        out_s = np.zeros((n, k), dtype=np.float32)
        all_i = torch.arange(self.n_items, device=self.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            ub = torch.from_numpy(uu[s:e]).to(self.device)
            c = e - s
            uu_rep = ub.repeat_interleave(self.n_items)
            ii_rep = all_i.repeat(c)
            sc = self.model(uu_rep, ii_rep).reshape(c, self.n_items).float().cpu().numpy()
            if R_train is not None:
                for r_i, u in enumerate(uu[s:e]):
                    a, b = R_train.indptr[u], R_train.indptr[u + 1]
                    sc[r_i, R_train.indices[a:b]] = -np.inf
            part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sc, part, axis=1)
            order = np.argsort(-vals, axis=1)
            out_i[s:e] = np.take_along_axis(part, order, axis=1)
            out_s[s:e] = np.take_along_axis(vals, order, axis=1)
        return out_i, out_s
