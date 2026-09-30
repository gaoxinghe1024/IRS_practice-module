"""List-level objective shared by greedy, MMR, GA and the exhaustive oracle.

    Rel(L)   = mean normalised relevance of the items in L
    Div(L)   = mean over unordered valid pairs of (1 - content_similarity); a pair is valid when the
               fixed similarity scorer (weighted_attributes v1) has >= 1 comparable dimension.
               No valid pair -> Div unavailable.
    Nov(i)   = 1 - log(1 + train_count_i) / log(1 + max_train_count); Nov(L) = mean.
               All counts zero -> Nov unavailable.
    Fresh(L) = share of items with >= 1 genre/theme tag not covered by the user's visible
               history; unknown tags are not "fresh"; empty history -> Fresh unavailable.
    F(L) = sum_t w_t * term_t / sum_t w_t over the available terms (weights redistributed).

The objective is set-level (order independent); the selected set is displayed
in descending relevance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_WEIGHTS = {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2, "fresh": 0.1}


@dataclass
class ListObjective:
    relevance: np.ndarray                 # [0, 1] per candidate
    sim: np.ndarray                       # (C, C) content similarity
    valid: np.ndarray                     # (C, C) bool, diagonal False
    novelty: np.ndarray | None            # [0, 1] per candidate, None = unavailable
    fresh: np.ndarray | None              # bool per candidate, None = unavailable
    weights: dict[str, float]

    @property
    def size(self) -> int:
        return len(self.relevance)

    def terms(self, pop: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorised terms for a population (P, K) of candidate positions; NaN = unavailable."""
        pop = np.atleast_2d(pop)
        P, K = pop.shape
        out = {"relevance": self.relevance[pop].mean(1)}
        if K >= 2:
            S = self.sim[pop[:, :, None], pop[:, None, :]]
            V = self.valid[pop[:, :, None], pop[:, None, :]]
            iu = np.triu_indices(K, 1)
            s, v = S[:, iu[0], iu[1]], V[:, iu[0], iu[1]]
            cnt = v.sum(1)
            div = np.where(cnt > 0, ((1 - s) * v).sum(1) / np.maximum(cnt, 1), np.nan)
        else:
            div = np.full(P, np.nan)
        out["diversity"] = div
        out["novelty"] = self.novelty[pop].mean(1) if self.novelty is not None else np.full(P, np.nan)
        out["fresh"] = self.fresh[pop].astype(float).mean(1) if self.fresh is not None else np.full(P, np.nan)
        return out

    def fitness(self, pop: np.ndarray) -> np.ndarray:
        t = self.terms(pop)
        num = np.zeros(len(t["relevance"]))
        den = np.zeros(len(t["relevance"]))
        for name, w in self.weights.items():
            val = t[name]
            ok = np.isfinite(val)
            num[ok] += w * val[ok]
            den[ok] += w
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def summary(self, sel: np.ndarray) -> dict:
        t = self.terms(np.asarray(sel)[None, :])
        return {"fitness": round(float(self.fitness(np.asarray(sel)[None, :])[0]), 5),
                **{k: (None if not np.isfinite(v[0]) else round(float(v[0]), 5)) for k, v in t.items()}}

    def available_terms(self) -> list[str]:
        out = ["relevance", "diversity"]
        if self.novelty is not None:
            out.append("novelty")
        if self.fresh is not None:
            out.append("fresh")
        return out


def novelty_scores(train_counts: np.ndarray, candidates: np.ndarray) -> np.ndarray | None:
    mx = float(np.max(train_counts)) if len(train_counts) else 0.0
    if mx <= 0:
        return None
    c = train_counts[candidates].astype(np.float64)
    return 1.0 - np.log1p(c) / np.log1p(mx)


def fresh_flags(snap, candidates: np.ndarray, history_tags: set[tuple[str, str]]) -> np.ndarray | None:
    if not history_tags:
        return None
    out = np.zeros(len(candidates), dtype=bool)
    for pos, i in enumerate(candidates):
        tags = [(ns, t) for ns in ("genre", "theme") if ns in snap.tags for t in snap.tags[ns].tags_of(int(i))]
        out[pos] = any(t not in history_tags for t in tags)
    return out
