import math

import numpy as np


def ranking_metrics(predictions, relevant, k=5):
    truth = set(relevant)
    if not truth:
        raise ValueError("An empty truth set cannot be scored")
    if k <= 0 or len(set(predictions)) != len(predictions):
        raise ValueError("Ranking must contain distinct IDs and k must be positive")
    hits = [int(aid in truth) for aid in predictions[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(truth))))
    return {"coverage_at_5": sum(hits) / len(truth), "hit_rate_at_5": float(any(hits)),
            "ndcg_at_5": dcg / ideal, "precision_at_5": sum(hits) / k}


def replacement_rule(baseline, candidate):
    delta = candidate["coverage_at_5"] - baseline["coverage_at_5"]
    ndcg_delta = candidate["ndcg_at_5"] - baseline["ndcg_at_5"]
    return dict(coverage_delta_pp=100 * delta, ndcg_delta=ndcg_delta,
                passes=bool(delta >= 0.02 - 1e-12 and ndcg_delta >= -1e-12))


def paired_bootstrap(left, right, seed=20260930, replicates=3000):
    delta = np.asarray(right) - np.asarray(left)
    if not len(delta):
        return None
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()
