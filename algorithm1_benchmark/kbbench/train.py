"""Weakly supervised logistic regression; co-liked pairs are NEVER evaluation truth."""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from .common import ROOT, read_rows, sha256, write_json, write_rows
from .features import DIMENSIONS, pair_features
from .prepare import SEED


def build_pairs(release, positive_count=50000):
    release = Path(release)
    cf = release / "data/cf"
    items = {item["anime_id"]: item for item in read_rows(ROOT / "data/catalog.jsonl.gz")}
    excluded = {row["anime_id"] for row in read_rows(ROOT / "data/targets.jsonl")}
    with (cf / "anime_mapping.csv").open(encoding="utf-8") as stream:
        mapping = list(csv.DictReader(stream))
    assert [int(row["anime_index"]) for row in mapping] == list(range(len(mapping)))
    item_ids = np.array([int(row["anime_id"]) for row in mapping])
    assert set(item_ids) == set(items)
    matrix = load_npz(cf / "ratings_train.npz").tocsr()
    assert matrix.shape[1] == len(item_ids)
    if matrix.nnz == 0 or not np.all((matrix.data >= 1) & (matrix.data <= 10)):
        raise ValueError("Expected uncentered train-only ratings")
    allowed = np.array([int(aid) not in excluded for aid in item_ids])
    rng = np.random.default_rng(SEED)
    positives = set()
    users_examined = 0
    # One pair per randomly visited user; prolific users cannot flood the training set.
    for user in rng.permutation(matrix.shape[0]):
        users_examined += 1
        start, end = matrix.indptr[user:user + 2]
        cols, values = matrix.indices[start:end], matrix.data[start:end]
        liked = cols[(values >= 8) & allowed[cols]]
        if len(liked) < 2:
            continue
        chosen = rng.choice(liked, size=2, replace=False)
        positives.add(tuple(sorted(int(aid) for aid in item_ids[chosen])))
        if len(positives) == positive_count:
            break
    if len(positives) < positive_count:
        raise ValueError(f"Only {len(positives)} distinct positives; requested {positive_count}")
    negatives = set()
    pool = np.array(sorted(set(items) - excluded))
    while len(negatives) < len(positives):
        pair = tuple(sorted(int(aid) for aid in rng.choice(pool, size=2, replace=False)))
        if pair not in positives:
            negatives.add(pair)
    rows = []
    for label, pairs in ((1, positives), (0, negatives)):
        for left, right in sorted(pairs):
            value = int.from_bytes(hashlib.sha256(f"{SEED}:{left}:{right}".encode()).digest()[:4], "big")
            rows.append(dict(left_id=left, right_id=right, label=label,
                             split="proxy_validation" if value % 5 == 0 else "train"))
    write_rows(ROOT / "data/training_pairs.jsonl.gz", rows)
    write_json(ROOT / "data/training_provenance.json", dict(
        seed=SEED, positive_pairs=len(positives), negative_pairs=len(negatives),
        users_examined=users_examined, users_in_matrix=matrix.shape[0], train_ratings=matrix.nnz,
        excluded_query_ids=sorted(excluded), excluded_from="both endpoints, positives and negatives",
        source_files={f"data/cf/{name}": sha256(cf / name) for name in ["ratings_train.npz", "anime_mapping.csv"]},
        positive_definition="Two distinct titles rated >=8 by one user, using ratings_train only; one sampled pair per visited user",
        negative_definition="Uniform random catalog pairs, excluding sampled positives and evaluation targets",
        limitations=["Unobserved random negatives may be false negatives", "Co-liked pairs express preference, not similarity truth", "No held-out ratings or MAL labels used for fitting"],
        output_sha256=sha256(ROOT / "data/training_pairs.jsonl.gz")))
    print(f"Prepared {len(rows)} weak training pairs; excluded all {len(excluded)} query IDs", flush=True)


def train():
    items = {item["anime_id"]: item for item in read_rows(ROOT / "data/catalog.jsonl.gz")}
    rows = list(read_rows(ROOT / "data/training_pairs.jsonl.gz"))
    excluded = {row["anime_id"] for row in read_rows(ROOT / "data/targets.jsonl")}
    assert all(row["left_id"] not in excluded and row["right_id"] not in excluded for row in rows)
    features = np.asarray([pair_features(items[row["left_id"]], items[row["right_id"]]) for row in rows])
    labels = np.array([row["label"] for row in rows])
    fit = np.array([row["split"] == "train" for row in rows])
    start = time.perf_counter()
    # No scaling: these are the same five [0,1] features as A. Fixed C, no MAL-driven tuning.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=SEED)
    model.fit(features[fit], labels[fit])
    if model.n_iter_[0] >= 2000:
        raise RuntimeError("Logistic regression did not converge")
    predicted = expit(features[~fit] @ model.coef_[0] + model.intercept_[0])
    result = dict(dimensions=list(DIMENSIONS), coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                  algorithm="L2 logistic regression, sklearn lbfgs", C=1.0, seed=SEED,
                  training_pairs=int(fit.sum()), proxy_validation_pairs=int((~fit).sum()),
                  proxy_validation_auc=float(roc_auc_score(labels[~fit], predicted)),
                  proxy_validation_log_loss=float(log_loss(labels[~fit], predicted)),
                  iterations=int(model.n_iter_[0]), fit_seconds=time.perf_counter() - start,
                  input_pairs_sha256=sha256(ROOT / "data/training_pairs.jsonl.gz"),
                  catalog_sha256=sha256(ROOT / "data/catalog.jsonl.gz"),
                  note="Proxy validation is not anime similarity ground truth, and is not used for replacement decisions")
    write_json(ROOT / "models/logistic.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, help="Only needed to regenerate weak pairs from frozen train CSR")
    parser.add_argument("--positive-count", type=int, default=50000)
    args = parser.parse_args()
    if args.release:
        build_pairs(args.release, args.positive_count)
    train()
