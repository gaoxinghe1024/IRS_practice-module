"""`tune` and `train` stages: A1 weight, GBDT (B0 / B1) and two-tower (C0) search, final seeds.

Training data: the shared sampled pairs of training-partition queries (1 positive : up to 4 weak
negatives, rule R0). Early stopping: binary log-loss / BCE on the validation-partition pairs (fixed,
built the same way). Trial selection: full-catalogue validation HR@8 on the primary set (R0, main
group, > 8 candidates), with the pre-registered tie-break. The test partition is never read here.

Process isolation: LightGBM (Homebrew OpenMP) and PyTorch (bundled OpenMP) must not share a process on
macOS, so the GBDT part and the two-tower part run as separate sub-processes (`*-gbdt`, `*-tower`) and
the parent stage merges their outputs. Neither half imports the other's library.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from app.recommendation.algorithm2.features import FEATURE_NAMES, FORMULA_FEATURES, compute_features
from app.recommendation.algorithm2.profiles import build_profile
from evaluation.algorithm2.common import mark_stage, now, read_json, sha256_file, write_json
from evaluation.algorithm2.data import load_prepared
from evaluation.algorithm2.evaluate import FeatureCache
from evaluation.algorithm2.metrics import summarize
from evaluation.algorithm2.split import load_pairs, load_queries

B_FEATURES = {"B0": FORMULA_FEATURES, "B1": FEATURE_NAMES}


# ------------------------------------------------------------------------------ shared inputs


class TrainingInputs:
    def __init__(self, run, log):
        self.cfg = run.config()
        self.prep = load_prepared(self.cfg)
        self.cat = self.prep.cat
        self.queries = load_queries(run)
        self.pairs = load_pairs(run)
        q = self.queries
        self.val_primary = q[(q.partition == "val") & (q.group == "main")].reset_index(drop=True)
        self.pair_queries = q[q.partition.isin(["train", "val"])].reset_index(drop=True)
        self._build_pairs(log)
        log(f"[inputs] pairs={len(self.pairs)} val primary queries={len(self.val_primary)}")
        self.val_cache = FeatureCache(self.cat, self.val_primary, log)

    def _build_pairs(self, log) -> None:
        t0 = time.time()
        row_of = {qid: r for r, qid in enumerate(self.pair_queries.query_id)}
        self.profiles = {}
        F = np.zeros((len(self.pairs), len(FEATURE_NAMES)), dtype=np.float32)
        qrow = np.zeros(len(self.pairs), dtype=np.int64)
        items = self.pairs.item.to_numpy()
        for qid, idx in self.pairs.groupby("query_id", sort=False).indices.items():
            r = row_of[qid]
            prof = build_profile(self.cat, np.asarray(self.pair_queries.visible.iat[r], dtype=np.int64))
            self.profiles[r] = prof
            F[idx] = compute_features(self.cat, prof, items[idx])
            qrow[idx] = r
        self.pair_F = F
        self.pair_qrow = qrow
        self.pair_y = self.pairs.label.to_numpy().astype(np.int32)
        self.is_train = (self.pairs.partition == "train").to_numpy()
        log(f"[inputs] pair features {F.shape} in {time.time() - t0:.0f}s")

    def gbdt_xy(self, family: str):
        idx = [FEATURE_NAMES.index(f) for f in B_FEATURES[family]]
        X = self.pair_F[:, idx]
        return X[self.is_train], self.pair_y[self.is_train], X[~self.is_train], self.pair_y[~self.is_train], idx


# ------------------------------------------------------------------------------ selection helpers


def pick_best(trials: list[dict], margin: float, complexity_key: str) -> dict:
    best_hr = max(t["val_HR@8"] for t in trials)
    close = [t for t in trials if t["val_HR@8"] >= best_hr - margin]
    close.sort(key=lambda t: (-t["val_NDCG@8"], t[complexity_key], t["trial"]))
    return close[0]


def deterministic_trials(grid: dict, n: int, seed: int) -> list[dict]:
    keys = sorted(grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]
    order = np.random.default_rng(seed).permutation(len(combos))
    return [combos[i] for i in order[:n]]


def _metrics_row(cache: FeatureCache, scores: np.ndarray) -> dict:
    m = cache.metrics_from_scores(scores)
    s = summarize([{"user": u, "hit": h, "ndcg": n, "mrr": r, "auc": a}
                   for u, h, n, r, a in m[["user", "hit", "ndcg", "mrr", "auc"]].itertuples(index=False)])
    return {"val_HR@8": s["HR@8"], "val_NDCG@8": s["NDCG@8"], "val_MRR@8": s["MRR@8"], "val_AUC": s["AUC"],
            "val_users": s["users"], "val_queries": s["queries"]}


# ------------------------------------------------------------------------------ GBDT (LightGBM, no torch)


def lgbm_params(params: dict, seed: int, gcfg: dict) -> dict:
    """LightGBM native parameters (same estimator as LGBMClassifier(objective='binary'))."""
    return {"objective": "binary", "learning_rate": float(params["learning_rate"]),
            "num_leaves": int(params["num_leaves"]), "max_depth": int(params["max_depth"]),
            "min_data_in_leaf": int(params["min_child_samples"]), "lambda_l2": float(params["reg_lambda"]),
            "seed": int(seed), "deterministic": True, "force_row_wise": True, "num_threads": int(gcfg["num_threads"]),
            "verbose": -1, "feature_pre_filter": False, "metric": "binary_logloss"}


def train_lgbm(Xtr, ytr, Xva, yva, params: dict, seed: int, gcfg: dict):
    import lightgbm as lgb

    evals: dict = {}
    t0 = time.time()
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    booster = lgb.train(lgbm_params(params, seed, gcfg), dtr, num_boost_round=int(gcfg["n_estimators"]),
                        valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(int(gcfg["early_stopping_rounds"]), verbose=False),
                                   lgb.record_evaluation(evals)])
    fit_s = time.time() - t0
    best_it = int(booster.best_iteration or booster.current_iteration())
    curve = {"train_logloss": [round(x, 6) for x in evals["train"]["binary_logloss"]],
             "val_logloss": [round(x, 6) for x in evals["val"]["binary_logloss"]], "best_iteration": best_it}
    return booster, best_it, curve, fit_s


def lgbm_val_scores(booster, cache: FeatureCache, idx: list[int], best_it: int, threads: int) -> np.ndarray:
    return np.asarray(booster.predict(cache.F[:, idx], raw_score=True, num_iteration=best_it, num_threads=threads),
                      dtype=np.float64)


def tune_gbdt(run, log) -> dict:
    cfg = run.config()
    ti = TrainingInputs(run, log)
    margin = float(cfg["selection"]["tie_margin"])
    trials, curves = [], {}
    a_trials = []
    for w in cfg["a1_weights"]:
        s = ti.val_cache.F[:, 0].astype(np.float64) + float(w) * ti.val_cache.F[:, 1].astype(np.float64)
        row = {"family": "A1", "trial": len(a_trials), "params": json.dumps({"hits_weight": w}), "hits_weight": w,
               **_metrics_row(ti.val_cache, s)}
        a_trials.append(row)
        log(f"[tune] A1 w={w}: HR@8={row['val_HR@8']:.4f} NDCG@8={row['val_NDCG@8']:.4f}")
    best_hr = max(t["val_HR@8"] for t in a_trials)
    close = sorted((t for t in a_trials if t["val_HR@8"] >= best_hr - margin),
                   key=lambda t: (-t["val_NDCG@8"], abs(t["hits_weight"] - 0.15), t["hits_weight"]))
    a1 = close[0]
    trials += a_trials
    gcfg = cfg["gbdt"]
    selected = {"A1": {"hits_weight": a1["hits_weight"], "val_HR@8": a1["val_HR@8"], "val_NDCG@8": a1["val_NDCG@8"]}}
    for fam in ("B0", "B1"):
        Xtr, ytr, Xva, yva, idx = ti.gbdt_xy(fam)
        fam_trials = []
        for t_i, params in enumerate(deterministic_trials(gcfg["grid"], int(gcfg["max_trials"]), int(gcfg["trial_seed"]))):
            booster, best_it, curve, fit_s = train_lgbm(Xtr, ytr, Xva, yva, params, 42, gcfg)
            t0 = time.time()
            sc = lgbm_val_scores(booster, ti.val_cache, idx, best_it, int(gcfg["num_threads"]))
            row = {"family": fam, "trial": t_i, "params": json.dumps(params), **params, "best_iteration": best_it,
                   "complexity": best_it * int(params["num_leaves"]), "fit_seconds": round(fit_s, 1),
                   "val_score_seconds": round(time.time() - t0, 1), "val_logloss_best": min(curve["val_logloss"]),
                   **_metrics_row(ti.val_cache, sc)}
            fam_trials.append(row)
            curves[f"{fam}_trial{t_i}"] = curve
            log(f"[tune] {fam} trial {t_i} {params} it={best_it}: HR@8={row['val_HR@8']:.4f} "
                f"NDCG@8={row['val_NDCG@8']:.4f} AUC={row['val_AUC']:.4f}")
        best = pick_best(fam_trials, margin, "complexity")
        selected[fam] = {"params": json.loads(best["params"]), "trial": best["trial"], "val_HR@8": best["val_HR@8"],
                         "features": B_FEATURES[fam]}
        trials += fam_trials
    out = {"selected": selected, "trials": trials, "curves": curves}
    write_json(run.work / "tuning_gbdt.json", out)
    return out


def train_gbdt(run, log) -> dict:
    cfg = run.config()
    tuning = read_json(run.path("tuning.json"))
    ti = TrainingInputs(run, log)
    gcfg = cfg["gbdt"]
    models, curves, importance = [], {}, []
    for fam in ("B0", "B1"):
        Xtr, ytr, Xva, yva, idx = ti.gbdt_xy(fam)
        params = tuning["selected"][fam]["params"]
        for seed in cfg["seeds"]:
            booster, best_it, curve, fit_s = train_lgbm(Xtr, ytr, Xva, yva, params, seed, gcfg)
            path = run.models / f"{fam}_seed{seed}.txt"
            booster.save_model(str(path), num_iteration=best_it)
            sc = lgbm_val_scores(booster, ti.val_cache, idx, best_it, int(gcfg["num_threads"]))
            gains = booster.feature_importance(importance_type="gain", iteration=best_it)
            for f, g in zip(B_FEATURES[fam], gains.tolist(), strict=True):
                importance.append({"family": fam, "seed": seed, "feature": f, "gain": round(float(g), 4)})
            models.append({"family": fam, "seed": seed, "file": f"models/{path.name}", "sha256": sha256_file(path),
                           "params": params, "features": B_FEATURES[fam], "best_iteration": best_it,
                           "fit_seconds": round(fit_s, 1), "bytes": path.stat().st_size, **_metrics_row(ti.val_cache, sc)})
            curves[f"{fam}_seed{seed}"] = curve
            log(f"[train] {fam} seed {seed}: it={best_it} val HR@8={models[-1]['val_HR@8']:.4f}")
    out = {"models": models, "curves": curves, "importance": importance, "catalog_hash": ti.cat.catalog_hash}
    write_json(run.work / "train_gbdt.json", out)
    return out


# ------------------------------------------------------------------------------ two-tower (PyTorch, no LightGBM)


def tower_inputs(ti: TrainingInputs, cfg: dict):
    from app.recommendation.algorithm2.two_tower import TowerEncoder

    enc = TowerEncoder(ti.cat, int(cfg["two_tower"]["studio_min_count"]))
    coll = np.stack([enc.collection_vector(ti.profiles[r]) for r in range(len(ti.pair_queries))]) \
        if len(ti.profiles) == len(ti.pair_queries) else np.stack(
        [enc.collection_vector(build_profile(ti.cat, np.asarray(v, dtype=np.int64))) for v in ti.pair_queries.visible])
    return enc, coll.astype(np.float32), enc.item_matrix()


def tower_val_scores(model, enc, ti: TrainingInputs, item_X: np.ndarray) -> np.ndarray:
    from app.recommendation.algorithm2.two_tower import collection_embeddings, item_embeddings

    emb = item_embeddings(model, item_X)
    vecs = np.stack([enc.collection_vector(build_profile(ti.cat, np.asarray(v, dtype=np.int64)))
                     for v in ti.val_primary.visible])
    cembs = collection_embeddings(model, vecs)
    out = np.zeros(len(ti.val_cache.cand), dtype=np.float64)
    for qi in range(len(ti.val_primary)):
        a, b = ti.val_cache.offsets[qi], ti.val_cache.offsets[qi + 1]
        out[a:b] = emb[ti.val_cache.cand[a:b]] @ cembs[qi]
    return out


def tower_pairs(ti: TrainingInputs):
    items = ti.pairs.item.to_numpy()
    tr = (ti.pair_qrow[ti.is_train], items[ti.is_train], ti.pair_y[ti.is_train])
    va = (ti.pair_qrow[~ti.is_train], items[~ti.is_train], ti.pair_y[~ti.is_train])
    return tr, va


def tower_cfg(cfg: dict, params: dict) -> dict:
    tc = cfg["two_tower"]
    return {**params, "hidden": tc["hidden"], "batch_size": tc["batch_size"], "max_epochs": tc["max_epochs"],
            "early_stopping_patience": tc["early_stopping_patience"]}


def _torch_setup(cfg: dict) -> None:
    import torch

    torch.set_num_threads(int(cfg["two_tower"]["torch_threads"]))


def tune_tower(run, log) -> dict:
    cfg = run.config()
    _torch_setup(cfg)
    from app.recommendation.algorithm2.two_tower import train_two_tower

    ti = TrainingInputs(run, log)
    tc = cfg["two_tower"]
    enc, coll, item_X = tower_inputs(ti, cfg)
    tr, va = tower_pairs(ti)
    trials, curves = [], {}
    for t_i, params in enumerate(deterministic_trials(tc["grid"], int(tc["max_trials"]), int(tc["trial_seed"]))):
        model, hist = train_two_tower(coll, item_X, tr, va, tower_cfg(cfg, params), 42, tc["device"], log)
        sc = tower_val_scores(model, enc, ti, item_X)
        row = {"family": "C0", "trial": t_i, "params": json.dumps(params), **params,
               "best_epoch": hist.get("best_epoch"), "epochs_run": len(hist["epoch"]),
               "complexity": int(sum(p.numel() for p in model.parameters())), "fit_seconds": hist["seconds"][-1],
               "val_logloss_best": hist["best_val_loss"], **_metrics_row(ti.val_cache, sc)}
        trials.append(row)
        curves[f"C0_trial{t_i}"] = hist
        log(f"[tune] C0 trial {t_i} {params} epochs={len(hist['epoch'])}: HR@8={row['val_HR@8']:.4f} "
            f"NDCG@8={row['val_NDCG@8']:.4f}")
    best = pick_best(trials, float(cfg["selection"]["tie_margin"]), "complexity")
    out = {"selected": {"C0": {"params": json.loads(best["params"]), "trial": best["trial"],
                               "val_HR@8": best["val_HR@8"]}}, "trials": trials, "curves": curves,
           "encoder": enc.spec()}
    write_json(run.work / "tuning_tower.json", out)
    return out


def train_tower(run, log) -> dict:
    cfg = run.config()
    _torch_setup(cfg)
    from app.recommendation.algorithm2.two_tower import save_model, train_two_tower

    tuning = read_json(run.path("tuning.json"))
    ti = TrainingInputs(run, log)
    tc = cfg["two_tower"]
    enc, coll, item_X = tower_inputs(ti, cfg)
    tr, va = tower_pairs(ti)
    params = tuning["selected"]["C0"]["params"]
    models, curves = [], {}
    for seed in cfg["seeds"]:
        tcfg = tower_cfg(cfg, params)
        model, hist = train_two_tower(coll, item_X, tr, va, tcfg, seed, tc["device"], log)
        path = run.models / f"C0_seed{seed}.pt"
        save_model(model, path, tcfg, enc.spec(), {"seed": seed, "catalog_hash": ti.cat.catalog_hash})
        sc = tower_val_scores(model, enc, ti, item_X)
        models.append({"family": "C0", "seed": seed, "file": f"models/{path.name}", "sha256": sha256_file(path),
                       "params": params, "best_epoch": hist.get("best_epoch"), "epochs_run": len(hist["epoch"]),
                       "fit_seconds": hist["seconds"][-1], "device": hist["device"], "bytes": path.stat().st_size,
                       "n_params": int(sum(p.numel() for p in model.parameters())), **_metrics_row(ti.val_cache, sc)})
        curves[f"C0_seed{seed}"] = hist
        log(f"[train] C0 seed {seed}: epochs={len(hist['epoch'])} val HR@8={models[-1]['val_HR@8']:.4f}")
    out = {"models": models, "curves": curves, "encoder": enc.spec()}
    write_json(run.work / "train_tower.json", out)
    return out


# ------------------------------------------------------------------------------ merging stages (parent process)


def tune(run, log, spawn) -> dict:
    if run.frozen():
        raise SystemExit("run is frozen: tuning after freeze is not allowed (create a new run instead)")
    spawn("tune-gbdt")
    spawn("tune-tower")
    g, t = read_json(run.work / "tuning_gbdt.json"), read_json(run.work / "tuning_tower.json")
    selected = {**g["selected"], **t["selected"]}
    pd.DataFrame(g["trials"] + t["trials"]).to_csv(run.path("trial_results.csv"), index=False)
    write_json(run.path("training_curves_tuning.json"), {**g["curves"], **t["curves"]})
    out = {"created_at": now(), "selected": selected, "encoder": t["encoder"],
           "note": "trial selection by full-catalogue validation HR@8 (primary set); early stopping by validation "
                   "pair log-loss"}
    write_json(run.path("tuning.json"), out)
    mark_stage(run.root, "tune", {"selected": {k: v.get("params", v.get("hits_weight")) for k, v in selected.items()}})
    return out


def train_final(run, log, spawn) -> dict:
    if run.frozen():
        raise SystemExit("run is frozen: retraining after freeze is not allowed")
    spawn("train-gbdt")
    spawn("train-tower")
    g, t = read_json(run.work / "train_gbdt.json"), read_json(run.work / "train_tower.json")
    tuning = read_json(run.path("tuning.json"))
    write_json(run.path("training_curves.json"), {**g["curves"], **t["curves"]})
    pd.DataFrame(g["importance"]).to_csv(run.path("feature_importance.csv"), index=False)
    out = {"created_at": now(), "models": g["models"] + t["models"], "a1_weight": tuning["selected"]["A1"]["hits_weight"],
           "encoder": t["encoder"], "catalog_hash": g["catalog_hash"]}
    write_json(run.path("models_manifest.json"), out)
    mark_stage(run.root, "train", {"models": len(out["models"])})
    return out
