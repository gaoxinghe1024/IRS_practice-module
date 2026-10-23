"""Algorithm 2 scorer B: gradient-boosted trees over collection-preference features.

Learns "does this candidate match this collection's taste" (hidden collection
item vs sampled unobserved items), not "does it hit a keyword". Uses scikit-
learn's HistGradientBoostingClassifier, which handles missing values natively;
explicit missingness masks are added as features as well.

Feature importance is permutation importance on the validation cases (global,
not a per-recommendation causal explanation).
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.profiles import CollectionProfile, affinity

FEATURES = ["genre_affinity", "theme_affinity", "studio_affinity", "demographic_affinity", "length_fit",
            "catalog_score", "train_popularity", "collection_size", "theme_missing", "studio_missing",
            "demographic_missing", "length_missing"]
SCORER_ID = "gradient_boosting_v1"


def collection_features(snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray) -> np.ndarray:
    vals, avail = affinity(snap, profile, candidates)
    pop = snap.train_interactions[candidates].astype(np.float64)
    maxpop = max(1.0, float(snap.train_interactions.max()))
    X = np.column_stack([
        vals["genre"], vals["theme"], vals["studio"], vals["demographic"], vals["length"],
        snap.score[candidates] / 10.0, np.log1p(pop) / np.log1p(maxpop),
        np.full(len(candidates), float(profile.size)),
        (~avail["theme"]).astype(float), (~avail["studio"]).astype(float),
        (~avail["demographic"]).astype(float), (~avail["length"]).astype(float),
    ])
    return X


class GBDTCollectionScorer:
    scorer_id = SCORER_ID
    version = "1.0"

    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta

    def score(self, snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray
              ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X = collection_features(snap, profile, candidates)
        p = self.model.predict_proba(X)[:, 1]
        return p, {"probability_uncalibrated": p}


def fit_gbdt(X_tr, y_tr, X_va, y_va, max_iter: int = 200, learning_rate: float = 0.05, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    best, best_auc = None, -1.0
    grid = [(learning_rate, 3), (learning_rate, 6), (learning_rate * 2, 3)]
    for lr, depth in grid:
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, max_depth=depth,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]) if len(set(y_va)) > 1 else float("nan")
        if auc > best_auc:
            best, best_auc, best_cfg = m, auc, {"learning_rate": lr, "max_depth": depth}
    imp = permutation_importance(best, X_va, y_va, scoring="roc_auc", n_repeats=5, random_state=seed)
    importance = sorted(({"feature": f, "importance": round(float(v), 5)}
                         for f, v in zip(FEATURES, imp.importances_mean, strict=True)), key=lambda d: -d["importance"])
    meta = {"scorer_id": SCORER_ID, "val_auc_sampled": round(float(best_auc), 4), "config": best_cfg,
            "features": FEATURES, "permutation_importance_val": importance,
            "negatives": "sampled_unobserved (same rule layer as positives)"}
    return best, meta
