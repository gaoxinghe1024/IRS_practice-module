"""Algorithm 2 scorers A: hand-written collection preference scoring.

manual_slides_v0      the slides' baseline, reproduced as written:
                          raw = catalog_score / 10 + 0.15 * tag_hits
                      tag_hits = number of the candidate's genre + theme tags that occur in the
                      collection profile. Its scale grows with vocabulary / tag counts, so scores
                      are not comparable across catalogues with different tag richness.
collection_profile_v1 normalised profile match (production default):
                          affinity_d(j) = sum_{t in tags_d(j)} profile_d(t) / max(1, |tags_d(j)|)
                          profile_match = availability-weighted mean of affinity_d and length fit
                          score = alpha * profile_match + (1 - alpha) * catalog_score / 10
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.profiles import CollectionProfile, affinity

DIM_WEIGHTS = {"genre": 0.35, "theme": 0.30, "demographic": 0.15, "studio": 0.10, "length": 0.10}


class ManualSlidesScorer:
    scorer_id = "manual_slides_v0"
    version = "0.1"

    def score(self, snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray
              ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        hits = np.zeros(len(candidates))
        for ns in ("genre", "theme"):
            tm = snap.tags.get(ns)
            f = profile.tag_freq.get(ns)
            if tm is None or f is None:
                continue
            present = (f > 0).astype(np.float32)
            hits += np.asarray(tm.X[candidates] @ present).ravel()
        quality = np.nan_to_num(snap.score[candidates], nan=0.0) / 10.0
        return quality + 0.15 * hits, {"tag_hits": hits, "quality": quality}


class CollectionProfileScorer:
    scorer_id = "collection_profile_v1"
    version = "1.0"

    def __init__(self, alpha: float = 0.7, weights: dict[str, float] | None = None):
        self.alpha = float(alpha)
        self.weights = {**DIM_WEIGHTS, **(weights or {})}

    def score(self, snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray
              ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        vals, avail = affinity(snap, profile, candidates)
        num = np.zeros(len(candidates))
        den = np.zeros(len(candidates))
        for d, w in self.weights.items():
            ok = avail[d]
            num[ok] += w * vals[d][ok]
            den[ok] += w
        match = np.divide(num, den, out=np.zeros(len(candidates)), where=den > 0)
        quality = np.nan_to_num(snap.score[candidates], nan=0.0) / 10.0
        comps = {f"{d}_affinity": vals[d] for d in self.weights}
        comps.update({"profile_match": match, "quality": quality, "coverage": den / sum(self.weights.values())})
        return self.alpha * match + (1 - self.alpha) * quality, comps
