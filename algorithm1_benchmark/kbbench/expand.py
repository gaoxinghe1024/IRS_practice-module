"""Freeze a larger, fresh cohort and a bounded optimization protocol before scoring."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT, read_rows, read_json, sha256, utc_now, write_json, write_rows

SEED = 20261001


def expand(release):
    protocol_path = ROOT / "data/optimization_protocol.json"
    if protocol_path.exists():
        raise ValueError("Cohort is already frozen; do not silently redraw it")
    history = ROOT / "history/v1"
    history.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "results", history / "results", dirs_exist_ok=True)
    for name in ["data/targets.jsonl", "data/mal_ground_truth.jsonl", "data/collection_audit.json",
                 "data/provenance.json", "data/training_provenance.json", "models/logistic.json"]:
        destination = history / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    old = list(read_rows(ROOT / "data/targets.jsonl"))
    excluded = {row["anime_id"] for row in old}
    items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
    ids = {row["anime_id"] for row in items}
    source = Path(release) / "data/canonical/anime_runtime.jsonl"
    popularity = {row["anime_id"]: int(row.get("members_count") or 0) for row in read_rows(source)}
    rest = sorted(ids - excluded, key=lambda aid: (popularity[aid], aid))
    rng = np.random.default_rng(SEED)
    new = []
    for quartile, group in enumerate(np.array_split(rest, 4), 1):
        new.extend(dict(anime_id=int(aid), sampling_group=f"random_q{quartile}") for aid in rng.choice(group, 75, replace=False))
    sampled = {row["anime_id"] for row in new}
    popular = sorted(ids - excluded, key=lambda aid: (-popularity[aid], aid))[:2000]
    pool = [aid for aid in popular if aid not in sampled]
    new.extend(dict(anime_id=int(aid), sampling_group="popular_supplement") for aid in rng.choice(pool, 300, replace=False))
    # Balanced 1/3 validation, 2/3 test within each sampling stratum; no labels consulted.
    for group in sorted({row["sampling_group"] for row in new}):
        members = sorted([row for row in new if row["sampling_group"] == group],
                         key=lambda row: hashlib.sha256(f"{SEED}:{row['anime_id']}".encode()).digest())
        for number, row in enumerate(members):
            row["split"] = "validation" if number < len(members) // 3 else "test"
    historical = [dict(row, original_split=row["split"], split="historical") for row in old]
    targets = historical + sorted(new, key=lambda row: row["anime_id"])
    assert len(targets) == len({row["anime_id"] for row in targets}) == 720
    assert len(new) == 600 and sum(row["split"] == "test" for row in new) == 400
    write_rows(ROOT / "data/targets.jsonl", targets)
    title_by_id = {row["anime_id"]: row["title"] for row in items}
    with (ROOT / "data/human_annotations_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target_id", "target_title", "similar_ids", "annotator", "rationale", "status"])
        writer.writeheader()
        for row in targets:
            writer.writerow(dict(target_id=row["anime_id"], target_title=title_by_id[row["anime_id"]], status="pending"))
    protocol = dict(version=2, frozen_at=utc_now(), seed=SEED, historical_targets=120, fresh_targets=600,
                    planned_validation=200, planned_test=400, targets_sha256=sha256(ROOT / "data/targets.jsonl"),
                    source_metadata_sha256=sha256(source),
                    sampling="300 uniform stratified titles (75 per membership quartile), plus 300 random titles from the top-2000 membership band; all exclude v1 targets",
                    allowed_inputs=["genres", "themes", "demographic", "studio", "episodes"],
                    A="unchanged repository weights and episode function",
                    B_candidates=["uniform_free", "uniform_monotone", "marginal_free", "marginal_monotone"],
                    B_fixed=dict(positive_count=50000, rating_threshold=8, C=1.0, features=5,
                                 marginal_negative="random endpoints drawn from positive-item frequency; reject self and sampled positive pairs",
                                 monotone="nonnegative five logistic coefficients; intercept free"),
                    C_candidates=[f"{text}_{center}" for text in ["original", "compact", "natural"] for center in ["raw", "centered"]],
                    C_fixed=dict(model="BAAI/bge-small-en-v1.5", dimensions=384, pooling="CLS", similarity="cosine",
                                 centering="optional unlabeled full-catalog mean subtraction followed by L2 normalization"),
                    D_candidates=["affine", "corpus_quantile"],
                    D_fixed=dict(tag_weight=0.6, semantic_weight=0.4, tag_path="A", semantic_path="validation-selected C",
                                 corpus_quantile="unlabeled random-pair cosine 1st/99th percentile linear clipping; no extra supervised training"),
                    selection="maximize 0.5*(validation coverage@5 + validation NDCG@5); stable candidate order breaks exact ties",
                    truth_policy="unchanged: top 5 MAL recommendations by votes/ID, minimum 3; no co-rating ground truth",
                    candidate_pool="all 14,202 titles except the query; no popularity/series filters",
                    test_policy="do not evaluate fresh test rankings until optimization_selection.json is written",
                    replacement_rule="test coverage at least +2pp relative to A AND test NDCG not lower",
                    forbidden=["synopsis", "title semantics", "user features at inference", "MAL recommendation edges as algorithm input", "changing K or truth definition to inflate scores"])
    write_json(protocol_path, protocol)
    metadata = read_json(ROOT / "data/provenance.json")
    metadata.update(original_target_count=metadata["target_count"], target_count=len(targets),
                    sampling=protocol["sampling"], optimization_protocol="data/optimization_protocol.json")
    write_json(ROOT / "data/provenance.json", metadata)
    print("Frozen 600 fresh targets (200 validation / 400 test) + 120 historical; protocol saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    expand(parser.parse_args().release)
