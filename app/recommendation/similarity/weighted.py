"""Algorithm 1 (A): weighted five-dimension attribute similarity.

For set-valued dimensions d in {genre, theme, demographic, studio}:
    J_d(i, j) = |A_d(i) & A_d(j)| / |A_d(i) | A_d(j)|
    available only when BOTH sides are non-empty (two empty sets are NOT similar).
Length: L(i, j) = exp(-|episodes_i - episodes_j| / tau), available when both counts are known.

    Sim_tag = sum_d w_d * avail_d * sim_d / sum_d w_d * avail_d
    coverage = sum_d w_d * avail_d / sum_d w_d

No available dimension -> insufficient_metadata (excluded from ranking, never "fully similar").
Score / popularity are not part of this formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.recommendation.features import SIM_NAMESPACES, CatalogSnapshot

SCORER_ID = "weighted_attributes"
SCORER_VERSION = "1.0"
DEFAULT_WEIGHTS = {"genre": 0.35, "theme": 0.30, "demographic": 0.15, "studio": 0.10, "length": 0.10}
DIMENSIONS = (*SIM_NAMESPACES, "length")


@dataclass
class SimilarityResult:
    candidates: np.ndarray          # catalogue indices
    score: np.ndarray               # NaN where insufficient metadata
    coverage: np.ndarray
    sims: dict[str, np.ndarray]     # per-dimension similarity (NaN where unavailable)
    available: dict[str, np.ndarray]

    def components(self, pos: int, weights: dict[str, float]) -> dict[str, dict]:
        out = {}
        denom = sum(weights[d] for d in DIMENSIONS if self.available[d][pos])
        for d in DIMENSIONS:
            if self.available[d][pos]:
                out[d] = {"similarity": round(float(self.sims[d][pos]), 4),
                          "weight": weights[d],
                          "contribution": round(float(weights[d] * self.sims[d][pos] / denom), 4) if denom else 0.0}
            else:
                out[d] = {"similarity": None, "weight": weights[d], "contribution": None, "missing": True}
        return out


class WeightedAttributeScorer:
    scorer_id = SCORER_ID
    version = SCORER_VERSION

    def __init__(self, weights: dict[str, float] | None = None, tau: float = 12.0):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.tau = float(tau)

    def score(self, snap: CatalogSnapshot, target: int, candidates: np.ndarray) -> SimilarityResult:
        candidates = np.asarray(candidates, dtype=np.int64)
        num = np.zeros(len(candidates))
        den = np.zeros(len(candidates))
        sims: dict[str, np.ndarray] = {}
        avail: dict[str, np.ndarray] = {}
        for ns in SIM_NAMESPACES:
            tm = snap.tags.get(ns)
            if tm is None:
                sims[ns] = np.full(len(candidates), np.nan)
                avail[ns] = np.zeros(len(candidates), dtype=bool)
                continue
            t_size = tm.sizes[target]
            c_size = tm.sizes[candidates]
            ok = (c_size > 0) & (t_size > 0)
            inter = np.asarray(tm.X[candidates] @ tm.X[target].T.toarray()).ravel() if t_size > 0 else \
                np.zeros(len(candidates))
            union = t_size + c_size - inter
            s = np.full(len(candidates), np.nan)
            s[ok] = inter[ok] / union[ok]
            sims[ns], avail[ns] = s, ok
            w = self.weights[ns]
            num[ok] += w * s[ok]
            den[ok] += w
        e_t = snap.episodes[target]
        e_c = snap.episodes[candidates]
        ok = np.isfinite(e_c) & np.isfinite(e_t)
        s = np.full(len(candidates), np.nan)
        s[ok] = np.exp(-np.abs(e_c[ok] - e_t) / self.tau)
        sims["length"], avail["length"] = s, ok
        num[ok] += self.weights["length"] * s[ok]
        den[ok] += self.weights["length"]
        total_w = sum(self.weights.values())
        score = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return SimilarityResult(candidates, score, den / total_w, sims, avail)

    def pairwise(self, snap: CatalogSnapshot, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Similarity matrix among a small list + validity mask (pairs with >= 1 comparable dimension)."""
        items = np.asarray(items, dtype=np.int64)
        k = len(items)
        num = np.zeros((k, k))
        den = np.zeros((k, k))
        for ns in SIM_NAMESPACES:
            tm = snap.tags.get(ns)
            if tm is None:
                continue
            Xs = tm.X[items]
            inter = (Xs @ Xs.T).toarray()
            sz = tm.sizes[items]
            union = sz[:, None] + sz[None, :] - inter
            ok = (sz[:, None] > 0) & (sz[None, :] > 0)
            s = np.divide(inter, union, out=np.zeros_like(inter), where=ok)
            num += np.where(ok, self.weights[ns] * s, 0)
            den += np.where(ok, self.weights[ns], 0)
        e = snap.episodes[items]
        ok = np.isfinite(e[:, None]) & np.isfinite(e[None, :])
        L = np.exp(-np.abs(np.nan_to_num(e[:, None]) - np.nan_to_num(e[None, :])) / self.tau)
        num += np.where(ok, self.weights["length"] * L, 0)
        den += np.where(ok, self.weights["length"], 0)
        valid = den > 0
        sim = np.divide(num, den, out=np.zeros_like(num), where=valid)
        np.fill_diagonal(valid, False)
        return sim, valid
