"""Metric definitions (see docs/evaluation-protocol.md).

For a query / user q with relevant set G_q and ranked list L_q:
    Hits@K    = 1 if L_q[:K] & G_q is non-empty else 0            (binary; NOT a coverage ratio)
    Recall@K  = |L_q[:K] & G_q| / |G_q|                            (coverage of the relevant set)
    MRR@K     = 1 / rank of the first relevant item within K, else 0
    NDCG@K    = DCG@K / IDCG@K, DCG = sum_r (2^rel_r - 1) / log2(r + 1)   (graded relevance)
Queries with an empty G_q are reported separately and never silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def hits_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return 1.0 if any(x in relevant for x in list(ranked)[:k]) else 0.0


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(list(ranked)[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    for pos, x in enumerate(list(ranked)[:k], start=1):
        if x in relevant:
            return 1.0 / pos
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum((2 ** g - 1) / math.log2(r + 1) for r, g in enumerate(gains, start=1))


def ndcg_at_k(gains_in_ranked_order: Sequence[float], ideal_gains_desc: Sequence[float], k: int) -> float:
    ideal = dcg(list(ideal_gains_desc)[:k])
    if ideal <= 0:
        return float("nan")
    return dcg(list(gains_in_ranked_order)[:k]) / ideal


def ndcg_binary(ranked: Sequence, relevant: set, k: int) -> float:
    gains = [1.0 if x in relevant else 0.0 for x in list(ranked)[:k]]
    return ndcg_at_k(gains, [1.0] * len(relevant), k)


def auc_pos_vs_neg(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """P(score_pos > score_neg) with ties counted 1/2 (Mann-Whitney)."""
    p = np.asarray(pos_scores, dtype=float)
    n = np.asarray(neg_scores, dtype=float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_pos = ranks[: len(p)].sum()
    return float((r_pos - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def mae(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true))) if len(true) else float("nan")


def rmse(pred, true) -> float:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2))) if len(true) else float("nan")


def intra_list_diversity(sim: np.ndarray, valid: np.ndarray) -> float:
    k = sim.shape[0]
    if k < 2:
        return float("nan")
    iu = np.triu_indices(k, 1)
    v = valid[iu]
    if not v.any():
        return float("nan")
    return float((1 - sim[iu])[v].mean())


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": round(float(v.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)], "n": int(len(v))}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean(a - b) with a 95% CI over paired cases (same users / queries)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) == 0:
        return {"mean_diff": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_diff": round(float(d.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "n": int(len(d))}
