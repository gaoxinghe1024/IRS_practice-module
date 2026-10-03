"""Generate and evaluate a reproducible 1,000-query synthetic benchmark.

The synthetic labels are generated from a declared metadata-only oracle before
any model rankings are computed.  They are useful for a controlled regression
benchmark, but they are not a substitute for user or expert annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from kbbench.common import ROOT, read_rows, sha256, write_json, write_rows
from kbbench.engine import METHODS, SimilarityEngine
from kbbench.features import DIMENSIONS, pair_features, rank_top_k
from kbbench.metrics import paired_bootstrap, ranking_metrics, replacement_rule

SEED = 20261003
QUERY_COUNT = 1000
K = 5
METHOD_NAMES = {
    "A": "Weighted five-dimension similarity",
    "B": "Logistic regression",
    "C": "BGE embeddings",
    "D": "0.6 tag + 0.4 semantic",
}


def stable_u64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def select_targets(ids: list[int]) -> list[int]:
    ranked = sorted(ids, key=lambda aid: (stable_u64(f"target:{SEED}:{aid}"), aid))
    return ranked[:QUERY_COUNT]


def oracle_labels(target_id: int, items: list[dict], index: dict[int, int], features: np.ndarray) -> tuple[list[int], str]:
    """Sample five labels from a metadata oracle with fixed per-target difficulty.

    The oracle uses a different weighting from A/D and adds fixed noise before
    sampling from one of three declared candidate pools.  The model rankings
    never enter this calculation.
    """
    rng = np.random.default_rng(stable_u64(f"oracle:{SEED}:{target_id}"))
    # Independent from A and D: studio is excluded and the oracle uses its own
    # rounded weights.  A small noise term preserves non-trivial examples.
    weights = np.asarray([0.50, 0.30, 0.10, 0.10], dtype=np.float64)
    base = features[:, [0, 1, 2, 4]] @ weights
    noise = rng.normal(0.0, 0.05, size=len(items))
    score = base + noise
    score[index[target_id]] = -np.inf
    order = np.argsort(-score, kind="stable")
    difficulty_draw = rng.random()
    if difficulty_draw < 0.40:
        pool_size, difficulty = 8, "easy"
    elif difficulty_draw < 0.80:
        pool_size, difficulty = 20, "medium"
    else:
        pool_size, difficulty = 60, "hard"
    pool = order[:pool_size]
    # Weighted sampling is fixed by the query seed and does not inspect results.
    logits = (score[pool] - np.max(score[pool])) / 0.035
    probability = np.exp(np.clip(logits, -30, 0))
    probability /= probability.sum()
    chosen = rng.choice(pool, size=K, replace=False, p=probability)
    relevant = sorted(int(items[position]["anime_id"]) for position in chosen)
    return relevant, difficulty


def evaluate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = SimilarityEngine(variant="optimized")
    items = engine.items
    index = engine.index
    targets = select_targets([int(aid) for aid in engine.ids])
    if len(targets) != QUERY_COUNT:
        raise ValueError(f"catalog has only {len(targets)} eligible targets")

    labels = []
    predictions = []
    per_query = []
    elapsed = []
    for target_id in targets:
        tick = time.perf_counter()
        scores, features = engine.scores(target_id)
        relevant, difficulty = oracle_labels(target_id, items, index, features)
        labels.append(dict(target_id=target_id, relevant_ids=relevant, source="synthetic_oracle",
                           split="test", sampling_group=difficulty))
        for method in METHODS:
            ranked = rank_top_k(engine.ids, scores[method], target_id, K)
            predicted_ids = [int(engine.ids[position]) for position in ranked]
            predictions.append(dict(target_id=target_id,
                                    target_title=items[index[target_id]]["title"], method=method,
                                    items=[dict(anime_id=int(engine.ids[position]),
                                                title=items[position]["title"],
                                                similarity=float(scores[method][position]),
                                                dimensions=dict(zip(DIMENSIONS, features[position].tolist())))
                                           for position in ranked]))
            per_query.append(dict(source="synthetic_oracle", split="test", method=method,
                                  target_id=target_id, sampling_group=difficulty,
                                  relevant_ids=relevant, predicted_ids=predicted_ids,
                                  **ranking_metrics(predicted_ids, relevant, K)))
        elapsed.append(time.perf_counter() - tick)

    grouped = defaultdict(list)
    for row in per_query:
        grouped[row["method"]].append(row)
    comparison = {method: dict(method=method, n_targets=len(rows), **{
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("coverage_at_5", "ndcg_at_5", "hit_rate_at_5", "precision_at_5")
    }) for method, rows in grouped.items()}
    decisions = {}
    base = sorted(grouped["A"], key=lambda row: row["target_id"])
    for method in ("B", "C", "D"):
        candidate = sorted(grouped[method], key=lambda row: row["target_id"])
        decision = replacement_rule(comparison["A"], comparison[method])
        decision["paired_95pct_delta_ci"] = {
            metric: paired_bootstrap([row[metric] for row in base], [row[metric] for row in candidate])
            for metric in ("coverage_at_5", "ndcg_at_5")
        }
        decision["evidence_scope"] = "synthetic_oracle"
        decision["automatic_system_replacement"] = False
        decisions[method] = decision

    strata = []
    for difficulty in ("easy", "medium", "hard"):
        for method in METHODS:
            rows = [row for row in per_query if row["sampling_group"] == difficulty and row["method"] == method]
            strata.append(dict(sampling_group=difficulty, method=method, n_targets=len(rows), **{
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("coverage_at_5", "ndcg_at_5", "hit_rate_at_5", "precision_at_5")
            }))

    manifest = dict(
        benchmark="algorithm1_synthetic_1000",
        seed=SEED,
        target_count=QUERY_COUNT,
        candidate_catalog_size=len(items),
        k=K,
        label_source="synthetic_metadata_oracle",
        label_policy=("For each target, score every other catalog item with "
                      "0.50 genre Jaccard + 0.30 theme Jaccard + 0.10 demographic exact "
                      "+ 0.10 length similarity, add seeded N(0, 0.05) noise, then "
                      "sample five without replacement from a fixed easy/medium/hard pool. "
                      "Studio is intentionally excluded from the oracle."),
        difficulty_policy="40% easy pool=8; 40% medium pool=20; 20% hard pool=60",
        target_selection="stable SHA-256 ordering of catalog IDs; no result-based filtering",
        posthoc_selection=False,
        model_selection_used=False,
        generator_sha256=sha256(Path(__file__)),
        numpy_version=np.__version__,
        limitations=["Synthetic labels are conditional on the declared metadata oracle.",
                     "The result is a regression/stress benchmark, not a real-user accuracy estimate.",
                     "Human or independent expert labels are still required for deployment decisions."],
        catalog_sha256=sha256(ROOT / "data/catalog.jsonl.gz"),
    )
    write_json(output / "testset_manifest.json", manifest)
    write_rows(output / "testset.jsonl", labels)
    write_rows(output / "ground_truth_used.jsonl", labels)
    write_rows(output / "predictions.jsonl", predictions)
    write_rows(output / "per_query_metrics.jsonl", per_query)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "split", "method", "n_targets",
                                                    "coverage_at_5", "ndcg_at_5", "hit_rate_at_5",
                                                    "precision_at_5"])
        writer.writeheader()
        for method in METHODS:
            writer.writerow(dict(source="synthetic_oracle", split="test", **comparison[method]))
    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="synthetic_oracle", variant=engine.variant, k=K,
        candidate_catalog_size=len(items), target_count=QUERY_COUNT,
        metric_definition={"coverage_at_5": "|Top5 intersect truth| / |truth| (Recall@5)",
                          "hit_rate_at_5": "1 if at least one Top5 hit else 0",
                          "ndcg_at_5": "binary relevance; ideal length min(5, truth size)",
                          "precision_at_5": "number of hits in Top5 / 5"},
        summaries=list(comparison.values()), strata=strata, replacement_decisions=decisions,
        synthetic_manifest_sha256=sha256(output / "testset_manifest.json"),
        inputs={"data/catalog.jsonl.gz": sha256(ROOT / "data/catalog.jsonl.gz"),
                "models/logistic_optimized.json": sha256(ROOT / "models/logistic_optimized.json"),
                "models/bge_optimized.npz": sha256(ROOT / "models/bge_optimized.npz")},
        timing={"all_four_methods_p50_ms": 1000 * statistics.median(elapsed),
                "all_four_methods_p95_ms": float(1000 * np.quantile(elapsed, 0.95)),
                "total_evaluation_seconds": time.perf_counter() - started},
        environment={"python": platform.python_version(), "platform": platform.platform()},
    )
    write_json(output / "summary.json", summary)
    print(json.dumps({method: comparison[method] for method in METHODS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic_1000")
    args = parser.parse_args()
    evaluate(args.output)
