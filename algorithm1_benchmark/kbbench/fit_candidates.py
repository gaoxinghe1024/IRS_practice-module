"""Four bounded logistic variants; labels remain co-high-rating versus random pairs."""
import hashlib
import time
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .common import ROOT, read_json, read_rows, sha256, write_json, write_rows
from .features import DIMENSIONS, pair_features
from .expand import SEED


def fit_monotone(features, labels, C=1.0):
    n = len(labels)
    def objective(params):
        w, intercept = params[:-1], params[-1]
        z = features @ w + intercept
        residual = expit(z) - labels
        loss = np.mean(np.logaddexp(0, z) - labels * z) + np.dot(w, w) / (2 * C * n)
        gradient = np.r_[features.T @ residual / n + w / (C * n), residual.mean()]
        return loss, gradient
    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True, method="L-BFGS-B",
                      bounds=[(0, None)] * features.shape[1] + [(None, None)],
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(f"Monotone logistic fit failed: {result.message}")
    return result.x[:-1], float(result.x[-1]), int(result.nit)


def make_marginal_pairs(rows):
    positives = {(row["left_id"], row["right_id"]) for row in rows if row["label"] == 1}
    frequencies = Counter(aid for pair in positives for aid in pair)
    ids = np.array(sorted(frequencies))
    weights = np.array([frequencies[int(aid)] for aid in ids], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(SEED)
    negatives = set()
    attempts = 0
    while len(negatives) < len(positives):
        # Batched draws preserve the endpoint marginal rather than favoring rare titles.
        for pair in rng.choice(ids, size=(10000, 2), p=weights):
            attempts += 1
            left, right = sorted(int(aid) for aid in pair)
            if left != right and (left, right) not in positives:
                negatives.add((left, right))
            if len(negatives) == len(positives):
                break
        if attempts > 5000000:
            raise ValueError("Could not sample enough distinct random negative pairs")
    result = []
    for label, pairs in ((1, positives), (0, negatives)):
        for left, right in sorted(pairs):
            # Use the original training seed for the pair split, independent of labels.
            digest = hashlib.sha256(f"20260930:{left}:{right}".encode()).digest()
            split = "proxy_validation" if int.from_bytes(digest[:4], "big") % 5 == 0 else "train"
            result.append(dict(left_id=left, right_id=right, label=label, split=split))
    return result


def fit_candidates():
    items = {row["anime_id"]: row for row in read_rows(ROOT / "data/catalog.jsonl.gz")}
    excluded = {row["anime_id"] for row in read_rows(ROOT / "data/targets.jsonl")}
    uniform = list(read_rows(ROOT / "data/training_pairs.jsonl.gz"))
    marginal = make_marginal_pairs(uniform)
    write_rows(ROOT / "data/training_pairs_marginal.jsonl.gz", marginal)
    candidates = {}
    for name, rows in (("uniform", uniform), ("marginal", marginal)):
        assert all(row["left_id"] not in excluded and row["right_id"] not in excluded for row in rows)
        x = np.array([pair_features(items[row["left_id"]], items[row["right_id"]]) for row in rows])
        y = np.array([row["label"] for row in rows])
        train = np.array([row["split"] == "train" for row in rows])
        path = ROOT / ("data/training_pairs.jsonl.gz" if name == "uniform" else "data/training_pairs_marginal.jsonl.gz")
        for constraint in ("free", "monotone"):
            started = time.perf_counter()
            if constraint == "free":
                model = LogisticRegression(C=1, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=SEED).fit(x[train], y[train])
                coef, intercept, iterations = model.coef_[0], float(model.intercept_[0]), int(model.n_iter_[0])
                if iterations >= 2000:
                    raise RuntimeError("Unconstrained fit did not converge")
            else:
                coef, intercept, iterations = fit_monotone(x[train], y[train])
            candidate = dict(dimensions=list(DIMENSIONS), coefficients=coef.tolist(), intercept=intercept,
                             negative_sampling=name, constraint=constraint, C=1.0, seed=SEED,
                             training_pairs=int(train.sum()), proxy_validation_pairs=int((~train).sum()),
                             proxy_validation_auc=float(roc_auc_score(y[~train], expit(x[~train] @ coef + intercept))),
                             iterations=iterations, fit_seconds=time.perf_counter() - started,
                             input_pairs_file=path.relative_to(ROOT).as_posix(), input_pairs_sha256=sha256(path),
                             catalog_sha256=sha256(ROOT / "data/catalog.jsonl.gz"), excluded_targets=len(excluded))
            candidates[f"{name}_{constraint}"] = candidate
            print(f"Fit {name}_{constraint}: {candidate['coefficients']}", flush=True)
    write_json(ROOT / "models/logistic_candidates.json", candidates)


if __name__ == "__main__":
    fit_candidates()
