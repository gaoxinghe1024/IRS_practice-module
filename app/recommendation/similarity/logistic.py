"""Algorithm 1 (B): logistic regression over the same five-dimension features.

Positives come only from similarity labels (human annotation, or a separately
declared weak source such as MAL user recommendations). Negatives are sampled
unlabelled catalogue items ("sampled_unlabelled" - not proven dissimilar).
Co-high-rating pairs are never used as similarity truth.

Queries are split into train / validation / test groups (franchise-connected
queries stay together). Standardisation and fitting use train only; the
regularisation strength is chosen on validation NDCG@5. The output is the
decision function - an uncalibrated score, not a similarity probability.
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.similarity.weighted import DIMENSIONS, SimilarityResult, WeightedAttributeScorer

SCORER_ID = "logistic_similarity"
FEATURES = [f"sim_{d}" for d in DIMENSIONS] + [f"avail_{d}" for d in DIMENSIONS] + ["coverage"]


def pair_features(res: SimilarityResult) -> np.ndarray:
    cols = [np.nan_to_num(res.sims[d], nan=0.0) for d in DIMENSIONS]
    cols += [res.available[d].astype(float) for d in DIMENSIONS]
    cols.append(res.coverage)
    return np.column_stack(cols)


class LogisticSimilarityScorer:
    scorer_id = SCORER_ID
    version = "1.0"

    def __init__(self, model, base: WeightedAttributeScorer, meta: dict):
        self.model = model
        self.base = base
        self.meta = meta

    def score(self, snap: CatalogSnapshot, target: int, candidates: np.ndarray) -> SimilarityResult:
        res = self.base.score(snap, target, candidates)
        s = self.model.decision_function(pair_features(res))
        s = np.where(np.isfinite(res.score), s, np.nan)   # keep insufficient-metadata semantics of A
        return SimilarityResult(res.candidates, s, res.coverage, res.sims, res.available)


def group_queries(label_rows: list[dict], components: dict[str, str], seed: int,
                  fractions=(0.6, 0.2, 0.2)) -> dict[str, str]:
    """Assign each query item to train/val/test by its franchise component."""
    from app.recommendation.collection.cases import hash_str
    from pipelines.processed_io import mix_hash

    out = {}
    for row in label_rows:
        q = row["query_item"]
        comp = components.get(q, q)
        h = float(mix_hash(np.array([hash_str(comp)], dtype=np.int64), seed)[0])
        out[q] = "train" if h < fractions[0] else ("val" if h < fractions[0] + fractions[1] else "test")
    return out


def training_pairs(snap: CatalogSnapshot, base: WeightedAttributeScorer, rows: list[dict], pool: np.ndarray,
                   negatives: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in rows:
        q = snap.index.get(row["query_item"])
        if q is None:
            continue
        pos = [snap.index[p["item_key"]] for p in row["positives"] if p["item_key"] in snap.index]
        if not pos:
            continue
        neg_pool = np.setdiff1d(pool, np.array(pos + [q]))
        neg = rng.choice(neg_pool, size=min(negatives, len(neg_pool)), replace=False)
        cand = np.array(pos + list(neg))
        res = base.score(snap, q, cand)
        ok = np.isfinite(res.score)
        X.append(pair_features(res)[ok])
        y.append(np.array([1] * len(pos) + [0] * len(neg))[ok])
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0)
    return np.vstack(X), np.concatenate(y)


def fit_logistic(snap: CatalogSnapshot, base: WeightedAttributeScorer, train_rows: list[dict], val_rows: list[dict],
                 pool: np.ndarray, negatives: int = 20, seed: int = 0, eval_fn=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X, y = training_pairs(snap, base, train_rows, pool, negatives, rng)
    if len(set(y.tolist())) < 2:
        return None, {"status": "untrained", "reason": "not enough labelled pairs"}
    best, best_score, trace = None, -np.inf, []
    for C in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(X, y)
        scorer = LogisticSimilarityScorer(model, base, {})
        val = eval_fn(scorer, val_rows) if eval_fn else float("nan")
        trace.append({"C": C, "val_ndcg@5": round(float(val), 4)})
        if val > best_score or best is None:
            best, best_score, best_c = model, val, C
    coefs = best[-1].coef_.ravel()
    meta = {"scorer_id": SCORER_ID, "status": "trained", "C": best_c, "selection": trace,
            "train_pairs": int(len(y)), "train_positive_pairs": int(y.sum()),
            "coefficients_standardised": {f: round(float(c), 4) for f, c in zip(FEATURES, coefs, strict=True)},
            "negatives": "sampled_unlabelled", "output": "decision_function (uncalibrated)"}
    return best, meta
