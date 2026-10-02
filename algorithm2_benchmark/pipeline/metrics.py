"""Metric definitions for algorithm 2 (one held-out target t per query, K = 8).

rank(t)   1 + #{scores > s_t} + #{scores == s_t and anime_id < anime_id_t}   (the served tie-break)
Hit@8     1 if rank(t) <= 8 else 0
NDCG@8    1 / log2(rank(t) + 1) if rank(t) <= 8 else 0   (single relevant item, IDCG = 1)
MRR@8     1 / rank(t) if rank(t) <= 8 else 0
AUC       (#{scores < s_t} + 0.5 * #{other candidates with score == s_t}) / (n - 1); None when n < 2.
          Raw-score ties count 0.5: the anime_id tie-break is not credited as discrimination.
Aggregation: per user mean over that user's queries, then mean over users (user-macro). AUC averages
only users with a defined value. Paired comparisons resample users (bootstrap) with replacement.
"""

from __future__ import annotations

import numpy as np

K = 8


def query_metrics(scores: np.ndarray, anime_ids: np.ndarray, target_pos: int, k: int = K) -> dict:
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    st = s[target_pos]
    higher = int((s > st).sum())
    ties = s == st
    tie_before = int((ties & (anime_ids < anime_ids[target_pos])).sum())
    rank = 1 + higher + tie_before
    n = len(s)
    hit = 1.0 if rank <= k else 0.0
    out = {"rank": rank, "hit": hit, "ndcg": (1.0 / np.log2(rank + 1)) if rank <= k else 0.0,
           "mrr": (1.0 / rank) if rank <= k else 0.0, "n_cand": n}
    if n >= 2:
        lower = int((s < st).sum())
        other_ties = int(ties.sum()) - 1
        out["auc"] = (lower + 0.5 * other_ties) / (n - 1)
    else:
        out["auc"] = None
    return out


def user_macro(users: np.ndarray, values: np.ndarray) -> tuple[float | None, int]:
    """Mean over users of each user's mean value (NaN / None values are ignored)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return None, 0
    u = np.asarray(users)[ok]
    v = v[ok]
    uniq, inv = np.unique(u, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    cnt = np.bincount(inv)
    return float((sums / cnt).mean()), int(len(uniq))


def per_user(users: np.ndarray, values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    u = np.asarray(users)[ok]
    uniq, inv = np.unique(u, return_inverse=True)
    sums = np.bincount(inv, weights=v[ok])
    cnt = np.bincount(inv)
    return dict(zip(uniq.tolist(), (sums / cnt).tolist(), strict=True))


def paired_bootstrap(a: dict, b: dict, reps: int = 2000, seed: int = 42) -> dict:
    """Mean of (a - b) over users present in both, with a user-level bootstrap 95% interval."""
    users = sorted(set(a) & set(b))
    if not users:
        return {"mean_diff": None, "ci95": [None, None], "users": 0}
    d = np.array([a[u] - b[u] for u in users])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "users": len(users),
            "share_boot_positive": float((boots > 0).mean())}


def summarize(rows: list[dict], key: str = "hit") -> dict:
    users = np.array([r["user"] for r in rows])
    out = {}
    for m in ("hit", "ndcg", "mrr", "auc"):
        vals = np.array([np.nan if r.get(m) is None else r[m] for r in rows], dtype=np.float64)
        mean, n_users = user_macro(users, vals)
        out[{"hit": "HR@8", "ndcg": "NDCG@8", "mrr": "MRR@8", "auc": "AUC"}[m]] = mean
        out[f"users_{m}"] = n_users
    out["queries"] = len(rows)
    out["users"] = int(len(set(users.tolist())))
    return out
