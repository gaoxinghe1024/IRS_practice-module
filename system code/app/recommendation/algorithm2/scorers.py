"""Algorithm-2 scorers behind one interface (offline evaluation and serving use the same objects).

| id     | family    | score                                                                          |
|--------|-----------|--------------------------------------------------------------------------------|
| A0     | formula   | rating_norm + 0.15 * hits (the slides' formula with the canonical hits)        |
| A1     | formula   | rating_norm + w * hits, w chosen on validation from [0.05, 0.10, 0.15, 0.20, 0.30] |
| B0     | lightgbm  | binary GBDT on [rating_norm, hits]; raw score (logit) used for ranking          |
| B1     | lightgbm  | binary GBDT on the eight features                                              |
| C0     | two_tower | dot product of collection and item tower outputs                               |
| AUX_slides_v0  | legacy | previous `manual_slides_v0` (presence-based hits), auxiliary reference      |
| AUX_profile_v1 | legacy | previous app default `collection_profile_v1`, auxiliary reference           |

Scores are only compared within a query; classifier outputs are not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.recommendation.algorithm2.catalog import Alg2Catalog
from app.recommendation.algorithm2.features import FEATURE_NAMES, compute_features
from app.recommendation.algorithm2.profiles import Alg2Profile

CORE_IDS = ("A0", "A1", "B0", "B1", "C0")
AUX_IDS = ("AUX_slides_v0", "AUX_profile_v1")
LABELS = {"A0": "Slides formula", "A1": "Tuned formula", "B0": "Gradient-boosted trees (2 features)",
          "B1": "Gradient-boosted trees (8 features)", "C0": "Two-tower model",
          "AUX_slides_v0": "Previous slides formula", "AUX_profile_v1": "Previous profile match"}


@dataclass
class QueryContext:
    cat: Alg2Catalog
    prof: Alg2Profile
    cand: np.ndarray
    _features: np.ndarray | None = field(default=None, repr=False)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = compute_features(self.cat, self.prof, self.cand)
        return self._features


class FormulaScorer:
    family = "formula"

    def __init__(self, scorer_id: str, weight: float):
        self.scorer_id, self.weight = scorer_id, float(weight)
        self.version = f"formula-w{self.weight:g}"

    def score(self, ctx: QueryContext) -> np.ndarray:
        F = ctx.features
        return F[:, 0].astype(np.float64) + self.weight * F[:, 1].astype(np.float64)

    def describe(self) -> dict:
        return {"scorer_id": self.scorer_id, "family": self.family, "hits_weight": self.weight}


class LightGBMScorer:
    family = "lightgbm"

    def __init__(self, scorer_id: str, booster, features: list[str], meta: dict | None = None, num_threads: int = 1):
        self.scorer_id, self.booster, self.features = scorer_id, booster, list(features)
        self.idx = [FEATURE_NAMES.index(f) for f in self.features]
        self.meta = meta or {}
        self.num_threads = int(num_threads)
        self.version = self.meta.get("model_sha256", "unsaved")[:12]

    def score_matrix(self, F: np.ndarray) -> np.ndarray:
        if len(F) == 0:
            return np.zeros(0)
        return np.asarray(self.booster.predict(F[:, self.idx], raw_score=True, num_threads=self.num_threads),
                          dtype=np.float64)

    def score(self, ctx: QueryContext) -> np.ndarray:
        return self.score_matrix(ctx.features)

    def describe(self) -> dict:
        return {"scorer_id": self.scorer_id, "family": self.family, "features": self.features, **self.meta}


class TreeEnsembleScorer(LightGBMScorer):
    """The same LightGBM model evaluated by the pure-NumPy reader (no native library; used in serving)."""

    family = "lightgbm"

    def __init__(self, scorer_id: str, ensemble, features: list[str], meta: dict | None = None):
        self.scorer_id, self.ensemble, self.features = scorer_id, ensemble, list(features)
        self.idx = [FEATURE_NAMES.index(f) for f in self.features]
        self.meta = {**(meta or {}), "engine": "numpy"}
        self.version = self.meta.get("model_sha256", "unsaved")[:12]
        ensemble.prepare()                           # evaluation tables at load, not on the first request

    def score_matrix(self, F: np.ndarray) -> np.ndarray:
        if len(F) == 0:
            return np.zeros(0)
        return self.ensemble.predict_raw(F[:, self.idx])


class TwoTowerScorer:
    family = "two_tower"

    def __init__(self, scorer_id: str, model, encoder, meta: dict | None = None):
        from app.recommendation.algorithm2.two_tower import item_embeddings

        self.scorer_id, self.model, self.encoder = scorer_id, model, encoder
        self.meta = meta or {}
        self.version = self.meta.get("model_sha256", "unsaved")[:12]
        self.item_emb = item_embeddings(model, encoder.item_matrix())      # per snapshot index

    def collection_embedding(self, prof: Alg2Profile) -> np.ndarray:
        from app.recommendation.algorithm2.two_tower import collection_embeddings

        return collection_embeddings(self.model, self.encoder.collection_vector(prof))[0]

    def score(self, ctx: QueryContext) -> np.ndarray:
        if len(ctx.cand) == 0:
            return np.zeros(0)
        c = self.collection_embedding(ctx.prof)
        return (self.item_emb[ctx.cand] @ c).astype(np.float64)

    def describe(self) -> dict:
        return {"scorer_id": self.scorer_id, "family": self.family, **self.meta}


class LegacyScorer:
    """Previous scorers, run exactly as before on the snapshot's own tags (auxiliary comparison only)."""

    family = "legacy"

    def __init__(self, scorer_id: str):
        from app.recommendation.collection.manual import CollectionProfileScorer, ManualSlidesScorer

        self.scorer_id = scorer_id
        self.impl = ManualSlidesScorer() if scorer_id == "AUX_slides_v0" else CollectionProfileScorer(0.7)
        self.version = self.impl.version

    def score(self, ctx: QueryContext) -> np.ndarray:
        from app.recommendation.profiles import build_collection_profile

        snap = ctx.cat.snap
        prof = build_collection_profile(snap, ctx.prof.history.tolist())
        s, _ = self.impl.score(snap, prof, ctx.cand)
        return np.asarray(s, dtype=np.float64)

    def describe(self) -> dict:
        return {"scorer_id": self.scorer_id, "family": self.family, "legacy_id": self.impl.scorer_id}


def rank_order(scores: np.ndarray, anime_id: np.ndarray) -> np.ndarray:
    """Positions sorted by score descending, ties by anime_id ascending (deterministic)."""
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    return np.lexsort((anime_id, -s))


def top_k(scores: np.ndarray, cand: np.ndarray, anime_id: np.ndarray, k: int = 8) -> np.ndarray:
    """Candidate positions of the Top-k (same tie-break as `rank_order`)."""
    if len(cand) == 0:
        return np.zeros(0, dtype=np.int64)
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    if len(cand) > 4 * k:
        part = np.argpartition(-s, k - 1)[:k]
        thr = s[part].min()
        pool = np.flatnonzero(s >= thr)                       # include every tie at the threshold
    else:
        pool = np.arange(len(cand))
    order = pool[np.lexsort((anime_id[cand[pool]], -s[pool]))]
    return order[:k]
