"""Select bounded variants using ONLY fresh validation labels, then freeze artifacts."""
import argparse
import hashlib
import json

import numpy as np
from scipy.special import expit

from .common import ROOT, read_json, read_rows, sha256, utc_now, write_json, write_rows
from .features import WEIGHTS, feature_matrix, rank_top_k, normalized_semantic
from .metrics import ranking_metrics
from .representations import CACHE, normalize_vectors
from .expand import SEED


def calibrate_cosines(vectors):
    rng = np.random.default_rng(SEED)
    left = rng.integers(0, len(vectors), size=20000)
    right = rng.integers(0, len(vectors) - 1, size=20000)
    right += right >= left
    cosines = np.sum(vectors[left] * vectors[right], axis=1)
    low, high = np.quantile(cosines, [0.01, 0.99])
    if high - low < 1e-8:
        raise ValueError("Degenerate corpus cosine range")
    return dict(low=float(low), high=float(high), seed=SEED, pairs=20000,
                pair_sha256=hashlib.sha256(np.stack([left, right], axis=1).astype("<i8").tobytes()).hexdigest(),
                source="unlabeled catalog vectors; no recommendation labels used")


def optimize(replay=False):
    selection_path = ROOT / "models/optimization_selection.json"
    if selection_path.exists() and not replay:
        raise ValueError("Selection is already frozen; do not silently retune after testing")
    frozen = read_json(selection_path) if replay else None
    protocol_path = ROOT / "data/optimization_protocol.json"
    protocol = read_json(protocol_path)
    if protocol["targets_sha256"] != sha256(ROOT / "data/targets.jsonl"):
        raise ValueError("Frozen target cohort changed")
    if replay and frozen["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Frozen optimization protocol changed")
    cohort = list(read_rows(ROOT / "data/targets.jsonl"))
    planned = {row["anime_id"] for row in cohort if row["split"] == "validation"}
    audit = read_json(ROOT / "data/collection_audit.json")
    if not planned <= {row["anime_id"] for row in audit}:
        raise ValueError("Finish collecting all planned validation targets before model selection")
    labels = [row for row in read_rows(ROOT / "data/mal_ground_truth.jsonl") if row["split"] == "validation"]
    if not labels or not {row["target_id"] for row in labels} <= planned:
        raise ValueError("Invalid validation cohort")
    labels.sort(key=lambda row: row["target_id"])
    if replay and labels != list(read_rows(ROOT / "results/selection_ground_truth.jsonl")):
        raise ValueError("Frozen validation labels changed")
    items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
    ids = np.array([row["anime_id"] for row in items])
    mapping = {int(aid): i for i, aid in enumerate(ids)}
    positions = [mapping[row["target_id"]] for row in labels]
    features = np.stack([feature_matrix(items[position], items) for position in positions])
    tag = np.zeros(features.shape[:2])
    for column, weight in enumerate(WEIGHTS):
        tag += features[:, :, column] * weight
    trials, per_query = [], []

    def assess(family, name, scores):
        metrics = []
        for index, label in enumerate(labels):
            order = rank_top_k(ids, scores[index], label["target_id"], 5)
            predicted = ids[order].tolist()
            values = ranking_metrics(predicted, label["relevant_ids"])
            metrics.append(values)
            per_query.append(dict(family=family, variant=name, target_id=label["target_id"], split="validation",
                                  predicted_ids=predicted, relevant_ids=label["relevant_ids"], **values))
        means = {metric: float(np.mean([row[metric] for row in metrics])) for metric in metrics[0]}
        trial = dict(family=family, variant=name, n_targets=len(labels), **means,
                     selection_utility=0.5 * (means["coverage_at_5"] + means["ndcg_at_5"]))
        trials.append(trial)
        print(f"Validation {family}/{name}: coverage={means['coverage_at_5']:.4f}, NDCG={means['ndcg_at_5']:.4f}", flush=True)
        return trial

    assess("A", "fixed", tag)
    logistic = read_json(ROOT / "models/logistic_candidates.json")
    b_trials = []
    for name in protocol["B_candidates"]:
        model = logistic[name]
        scores = expit(features @ np.asarray(model["coefficients"]) + model["intercept"])
        b_trials.append(assess("B", name, scores))
    chosen_b = max(b_trials, key=lambda trial: trial["selection_utility"])["variant"]
    c_trials = []
    for name in protocol["C_candidates"]:
        style, normalization = name.rsplit("_", 1)
        path = ROOT / "models/bge_embeddings.npz" if style == "original" else CACHE / f"{style}.npz"
        with np.load(path, allow_pickle=False) as artifact:
            if not np.array_equal(artifact["anime_ids"], ids):
                raise ValueError("Representation ID mapping mismatch")
            vectors, _ = normalize_vectors(artifact["vectors"], centered=normalization == "centered")
        cosines = np.clip(vectors[positions] @ vectors.T, -1, 1)
        c_trials.append(assess("C", name, cosines))
    chosen_c = max(c_trials, key=lambda trial: trial["selection_utility"])["variant"]
    style, normalization = chosen_c.rsplit("_", 1)
    path = ROOT / "models/bge_embeddings.npz" if style == "original" else CACHE / f"{style}.npz"
    with np.load(path, allow_pickle=False) as artifact:
        vectors, mean = normalize_vectors(artifact["vectors"], centered=normalization == "centered")
    cosines = np.clip(vectors[positions] @ vectors.T, -1, 1)
    calibration = calibrate_cosines(vectors)
    d_trials = [assess("D", name, 0.6 * tag + 0.4 * normalized_semantic(cosines, name, calibration)) for name in protocol["D_candidates"]]
    chosen_d = max(d_trials, key=lambda trial: trial["selection_utility"])["variant"]
    if replay:
        if (chosen_b, chosen_c, chosen_d) != (frozen["B"], frozen["C"], frozen["D"]):
            raise ValueError("Validation replay disagrees with the frozen selection")
        saved_b = read_json(ROOT / "models/logistic_optimized.json")
        np.testing.assert_allclose(logistic[chosen_b]["coefficients"], saved_b["coefficients"], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(logistic[chosen_b]["intercept"], saved_b["intercept"], rtol=1e-5, atol=1e-7)
        with np.load(ROOT / "models/bge_optimized.npz", allow_pickle=False) as saved_c:
            np.testing.assert_array_equal(ids, saved_c["anime_ids"])
            np.testing.assert_allclose(vectors, saved_c["vectors"], rtol=1e-5, atol=1e-6)
        for bound in ("low", "high"):
            np.testing.assert_allclose(calibration[bound], frozen["semantic_calibration"][bound], rtol=1e-5, atol=1e-7)
        print("Validation replay matches frozen B/C/D; no selection or artifact was overwritten", flush=True)
        return
    write_json(ROOT / "models/logistic_optimized.json", logistic[chosen_b])
    np.savez_compressed(ROOT / "models/bge_optimized.npz", anime_ids=ids, vectors=vectors)
    c_meta = read_json(ROOT / "models/bge_metadata.json" if style == "original" else CACHE / f"{style}.json")
    c_meta.update(text_style=style, normalization=normalization, centering_mean=mean.tolist(),
                  vector_sha256=sha256(ROOT / "models/bge_optimized.npz"),
                  note="Original 384-dim BGE vectors optionally centered using unlabeled full-catalog mean; cosine remains the similarity")
    write_json(ROOT / "models/bge_optimized_metadata.json", c_meta)
    write_json(ROOT / "results/validation_trials.json", trials)
    write_rows(ROOT / "results/validation_trial_metrics.jsonl", per_query)
    write_rows(ROOT / "results/selection_ground_truth.jsonl", labels)
    # Freeze LAST, only after selected artifacts and validation evidence exist.
    selection = dict(frozen_at=utc_now(), protocol_sha256=sha256(protocol_path),
                     validation_target_ids=[row["target_id"] for row in labels], validation_n=len(labels),
                     validation_truth_sha256=sha256(ROOT / "results/selection_ground_truth.jsonl"),
                     baseline_A_unchanged=True, B=chosen_b, C=chosen_c, D=chosen_d,
                     semantic_calibration=calibration, selection_objective=protocol["selection"],
                     test_rankings_used_for_selection=False,
                     files={name: sha256(ROOT / name) for name in ["models/logistic_optimized.json", "models/bge_optimized.npz", "models/bge_optimized_metadata.json", "results/validation_trials.json"]})
    write_json(ROOT / "models/optimization_selection.json", selection)
    print(f"FROZEN B={chosen_b}, C={chosen_c}, D={chosen_d}; validation n={len(labels)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Reproduce validation selection without overwriting frozen artifacts")
    optimize(replay=parser.parse_args().replay)
