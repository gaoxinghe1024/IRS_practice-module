import argparse
import csv
import importlib.metadata
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, read_json, read_rows, sha256, utc_now, write_json, write_rows
from .engine import METHODS, SimilarityEngine
from .features import DIMENSIONS, rank_top_k
from .metrics import paired_bootstrap, ranking_metrics, replacement_rule

METRICS = ("coverage_at_5", "ndcg_at_5", "hit_rate_at_5", "precision_at_5")


def human_labels(path, catalog, cohort):
    labels, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"].strip() != "verified":
                continue
            target = int(row["target_id"])
            relevant = [int(value.strip()) for value in row["similar_ids"].split(";") if value.strip()]
            if target not in cohort or target in seen:
                raise ValueError("Human targets must be unique members of the frozen cohort")
            if len(relevant) not in (3, 4, 5) or len(set(relevant)) != len(relevant):
                raise ValueError("Each verified target needs 3-5 distinct similar IDs")
            if target in relevant or not set(relevant) <= catalog or not row["annotator"].strip() or not row["rationale"].strip():
                raise ValueError("Invalid human IDs or missing annotator/rationale")
            seen.add(target)
            labels.append(dict(target_id=target, relevant_ids=relevant, source="human",
                               split=cohort[target]["split"], sampling_group=cohort[target]["sampling_group"]))
    if not labels:
        raise ValueError("No verified human annotations; pending rows are not ground truth")
    return labels


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["split"], row["method"])].append(row)
    result = []
    for (source, split, method), entries in sorted(grouped.items()):
        result.append(dict(source=source, split=split, method=method, n_targets=len(entries),
                           **{metric: float(np.mean([row[metric] for row in entries])) for metric in METRICS}))
    return result


def evaluate(human=None, output=None):
    output = Path(output) if output else ROOT / ("results/human" if human else "results")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = SimilarityEngine()
    loaded_seconds = time.perf_counter() - started
    cohort = {row["anime_id"]: row for row in read_rows(ROOT / "data/targets.jsonl")}
    if human:
        labels = human_labels(human, set(engine.index), cohort)
    else:
        labels = list(read_rows(ROOT / "data/mal_ground_truth.jsonl"))
    primary_source = "human" if human else "mal_recommendations"
    if not labels:
        raise ValueError("No similarity ground truth; cannot publish comparison metrics")
    if not any(row["split"] == "test" for row in labels):
        raise ValueError("No test labels available")
    franchise = defaultdict(set)
    for row in read_rows(ROOT / "data/franchise_links.jsonl.gz"):
        if row["target_id"] in cohort:
            franchise[row["target_id"]].add(row["similar_id"])
    labels += [dict(target_id=aid, relevant_ids=sorted(relevant), source="franchise_sanity",
                    split=cohort[aid]["split"], sampling_group=cohort[aid]["sampling_group"])
               for aid, relevant in sorted(franchise.items())]
    targets = sorted({row["target_id"] for row in labels})
    label_keys = [(row["source"], row["target_id"]) for row in labels]
    if len(set(label_keys)) != len(label_keys):
        raise ValueError("Repeated ground-truth query")
    # Recheck exclusions at evaluation time, including custom human ground truth.
    training = list(read_rows(ROOT / "data/training_pairs.jsonl.gz"))
    assert not set(cohort) & {row[key] for row in training for key in ("left_id", "right_id")}
    predictions, prediction_map, elapsed = [], {}, []
    for aid in targets:
        tick = time.perf_counter()
        scores, features = engine.scores(aid)
        for method in METHODS:
            ranked = rank_top_k(engine.ids, scores[method], aid, 5)
            output_items = [dict(anime_id=int(engine.ids[position]), title=engine.items[position]["title"],
                                 similarity=float(scores[method][position]),
                                 dimensions=dict(zip(DIMENSIONS, features[position].tolist()))) for position in ranked]
            predicted_ids = [item["anime_id"] for item in output_items]
            assert aid not in predicted_ids and len(set(predicted_ids)) == 5
            prediction_map[(aid, method)] = predicted_ids
            predictions.append(dict(target_id=aid, target_title=engine.items[engine.index[aid]]["title"],
                                    method=method, items=output_items))
        elapsed.append(time.perf_counter() - tick)
    per_query = []
    for label in labels:
        relevant = label["relevant_ids"]
        if not relevant or not set(relevant) <= set(engine.index) or label["target_id"] in relevant:
            raise ValueError("Invalid ground truth")
        for method in METHODS:
            predicted = prediction_map[(label["target_id"], method)]
            per_query.append(dict(source=label["source"], split=label["split"], method=method,
                                  target_id=label["target_id"], sampling_group=label["sampling_group"],
                                  relevant_ids=relevant, predicted_ids=predicted,
                                  **ranking_metrics(predicted, relevant)))
    summaries = summarize(per_query)
    comparison = {row["method"]: row for row in summaries if row["source"] == primary_source and row["split"] == "test"}
    decisions = {}
    for method in ("B", "C", "D"):
        base = sorted([row for row in per_query if row["source"] == primary_source and row["split"] == "test" and row["method"] == "A"], key=lambda row: row["target_id"])
        candidate = sorted([row for row in per_query if row["source"] == primary_source and row["split"] == "test" and row["method"] == method], key=lambda row: row["target_id"])
        assert [row["target_id"] for row in base] == [row["target_id"] for row in candidate]
        decision = replacement_rule(comparison["A"], comparison[method])
        decision["paired_95pct_delta_ci"] = {metric: paired_bootstrap([row[metric] for row in base], [row[metric] for row in candidate]) for metric in ("coverage_at_5", "ndcg_at_5")}
        decision["evidence_scope"] = primary_source
        decision["automatic_system_replacement"] = False
        decisions[method] = decision
    # Separate anchor/random summaries make the popular-title bias visible.
    strata = []
    for group in sorted({row["sampling_group"] for row in per_query}):
        for summary in summarize([row for row in per_query if row["sampling_group"] == group]):
            strata.append(dict(sampling_group=group, **summary))
    environment = dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                       packages={package: importlib.metadata.version(package) for package in ["numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "matplotlib"]})
    summary = dict(created_at=utc_now(), source=primary_source, k=5, candidate_catalog_size=len(engine.items),
                   candidate_policy="Every catalog title except the target; no sampled candidate pool", target_count=len(targets),
                   metric_definition={"coverage_at_5": "slide HR@5 = |Top5 intersect truth| / |truth| (standard Recall@5)",
                                      "hit_rate_at_5": "1 if at least one Top5 hit else 0", "ndcg_at_5": "binary relevance; ideal length min(5, truth size)"},
                   summaries=summaries, strata=strata, replacement_decisions=decisions,
                   human_validation="completed_for_supplied_verified_rows" if human else "pending; user authorized MAL evaluation first",
                   environment=environment,
                   timing=dict(load_seconds=loaded_seconds, all_four_methods_p50_ms=1000 * statistics.median(elapsed),
                               all_four_methods_p95_ms=float(1000 * np.quantile(elapsed, 0.95)), total_evaluation_seconds=time.perf_counter()-started),
                   inputs={str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in [ROOT / "data/catalog.jsonl.gz", ROOT / "data/targets.jsonl", ROOT / "models/logistic.json", ROOT / "models/bge_metadata.json", ROOT / "data/training_pairs.jsonl.gz"]})
    label_path = Path(human) if human else ROOT / "data/mal_ground_truth.jsonl"
    summary["ground_truth_sha256"] = sha256(label_path)
    write_json(output / "summary.json", summary)
    write_rows(output / "ground_truth_used.jsonl", labels)
    write_rows(output / "predictions.jsonl", predictions)
    write_rows(output / "per_query_metrics.jsonl", per_query)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source", "split", "method", "n_targets", *METRICS])
        writer.writeheader()
        writer.writerows(summaries)
    print({key: comparison[key] for key in METHODS}, flush=True)
    print("Replacement decisions:", decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.human, args.output)
