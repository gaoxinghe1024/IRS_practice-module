"""Route A selected in algorithm1_benchmark and the final report, section 7.1.

Genre/theme use Jaccard (missing -> 0). Demographic/studio use a scalar,
case-insensitive exact match (missing -> 0). The catalogue stores these as
sets, so the alphabetically first normalised value is the primary value.
Length is 1 / (1 + abs(e1 - e2) / 12), with 0.5 for missing/nonpositive counts.
Weights are fixed; metadata coverage is diagnostic and never changes the score.
The original weighted scorer remains separate for existing models/rerankers.
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.similarity.weighted import DEFAULT_WEIGHTS, DIMENSIONS, SimilarityResult

SCORER_ID = "algorithm1_a"
SCORER_VERSION = "1.0"


class Algorithm1Scorer:
    scorer_id = SCORER_ID
    version = SCORER_VERSION
    fixed_weights = True

    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)

    def score(self, snap: CatalogSnapshot, target: int, candidates: np.ndarray) -> SimilarityResult:
        candidates = np.asarray(candidates, dtype=np.int64)
        sims, available = {}, {}
        for ns in DIMENSIONS[:-1]:
            tm = snap.tags.get(ns)
            ok = np.zeros(len(candidates), dtype=bool)
            values = np.zeros(len(candidates), dtype=np.float64)
            if tm is not None:
                ok = (tm.sizes[target] > 0) & (tm.sizes[candidates] > 0)
                if ok.any():
                    if ns in ("genre", "theme"):
                        inter = np.asarray(tm.X[candidates] @ tm.X[target].T.toarray()).ravel().astype(np.float64)
                        union = tm.sizes[target] + tm.sizes[candidates].astype(np.float64) - inter
                        values[ok] = inter[ok] / union[ok]
                    else:
                        # build_snapshot sorts vocabulary by normalised label.
                        primary = tm.cols_of(target)[0]
                        values[ok] = [float(tm.cols_of(int(i))[0] == primary) for i in candidates[ok]]
            sims[ns], available[ns] = values, ok

        target_eps, candidate_eps = snap.episodes[target], snap.episodes[candidates]
        ok = np.isfinite(target_eps) & (target_eps > 0) & np.isfinite(candidate_eps) & (candidate_eps > 0)
        length = np.full(len(candidates), 0.5, dtype=np.float64)
        length[ok] = 1.0 / (1.0 + np.abs(candidate_eps[ok] - target_eps) / 12.0)
        sims["length"], available["length"] = length, ok

        score, coverage = np.zeros(len(candidates)), np.zeros(len(candidates))
        for ns in DIMENSIONS:
            score += self.weights[ns] * sims[ns]
            coverage += self.weights[ns] * available[ns]
        return SimilarityResult(candidates, score, coverage, sims, available)

    @staticmethod
    def rank(scores: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Descending score, numeric anime ID ascending; key resolves equal IDs."""
        ids = np.array([int(str(key).rsplit(":", 1)[-1]) if str(key).rsplit(":", 1)[-1].isdigit() else i
                        for i, key in enumerate(keys)], dtype=np.int64)
        return np.lexsort((keys, ids, -scores))
