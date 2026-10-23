"""`freeze` stage and the test-time decision (rules come from selection_protocol.json, written at split).

freeze: pick the candidate to deploy on validation only (A1, B0, B1, C0), name the conservative fallback
(A1), and record every artefact the test evaluation and the service will load, with SHA-256 hashes.
decide: apply the pre-registered confirmation rules to the frozen test results; never switch to a model
because it happened to win on test.
"""

from __future__ import annotations

import numpy as np

from algorithm2_benchmark.pipeline.common import (
    git_state,
    mark_stage,
    now,
    read_json,
    sha256_file,
    write_json,
)
from app.recommendation.algorithm2.features import FEATURE_VERSION, feature_schema

COMPLEXITY_ORDER = {"A1": 0, "B0": 1, "B1": 2, "C0": 3}
FEATURES_OF = {"B0": ["rating_norm", "hits"], "B1": None, "C0": None}


def _entropy_threshold(run, cfg) -> float | None:
    from algorithm2_benchmark.pipeline.data import load_prepared
    from algorithm2_benchmark.pipeline.final_eval import entropy_threshold
    from algorithm2_benchmark.pipeline.split import load_queries

    q = load_queries(run)
    return entropy_threshold(load_prepared(cfg).cat, q[q.partition == "val"])


def freeze(run, log) -> dict:
    if run.frozen():
        log("[freeze] already frozen; nothing to do")
        return read_json(run.path("freeze_manifest.json"))
    cfg = run.config()
    protocol = read_json(run.path("selection_protocol.json"))
    tuning = read_json(run.path("tuning.json"))
    mm = read_json(run.path("models_manifest.json"))
    margin = float(cfg["selection"]["tie_margin"])
    fams = {"A1": {"val_HR@8": tuning["selected"]["A1"]["val_HR@8"], "val_NDCG@8": None, "seeds": None}}
    trials = {r["family"]: r for r in _trial_rows(run) if r["family"] == "A1" and
              abs(float(r["hits_weight"]) - float(mm["a1_weight"])) < 1e-12}
    if "A1" in trials:
        fams["A1"]["val_NDCG@8"] = float(trials["A1"]["val_NDCG@8"])
    for fam in ("B0", "B1", "C0"):
        rows = [m for m in mm["models"] if m["family"] == fam]
        fams[fam] = {"val_HR@8": float(np.mean([m["val_HR@8"] for m in rows])),
                     "val_HR@8_sd": float(np.std([m["val_HR@8"] for m in rows])),
                     "val_NDCG@8": float(np.mean([m["val_NDCG@8"] for m in rows])),
                     "seeds": {m["seed"]: m["val_HR@8"] for m in rows}}
    best_hr = max(v["val_HR@8"] for v in fams.values())
    close = [f for f, v in fams.items() if v["val_HR@8"] >= best_hr - margin]
    close.sort(key=lambda f: (-(fams[f]["val_NDCG@8"] or 0.0), COMPLEXITY_ORDER[f]))
    pre = close[0]
    deploy_seed = int(cfg["deploy_seed"])
    artefacts = []
    for m in mm["models"]:
        p = run.root / m["file"]
        actual = sha256_file(p)
        if actual != m["sha256"]:
            raise SystemExit(f"{m['file']} changed after training (hash mismatch); refusing to freeze")
        artefacts.append({**{k: m[k] for k in ("family", "seed", "file", "sha256")}, "bytes": p.stat().st_size})
    equivalence = _gbdt_equivalence(run, cfg, mm["models"])
    if not equivalence["equal_within_1e-9"]:
        raise SystemExit(f"NumPy GBDT reader differs from native LightGBM: {equivalence}")
    manifest = {
        "frozen_at": now(), "run_id": cfg["run_id"], "level": cfg["level"], "git": git_state(),
        "gbdt_serving_engine": {"engine": "numpy reader of the LightGBM text model (app/recommendation/algorithm2/"
                                          "gbdt_tree.py)", "check_vs_native": equivalence},
        "catalog_hash": mm["catalog_hash"], "feature_schema": feature_schema(), "feature_version": FEATURE_VERSION,
        "encoder": mm["encoder"],
        "formula": {"A0": {"hits_weight": 0.15}, "A1": {"hits_weight": mm["a1_weight"]}},
        "models": artefacts,
        "configs": {f: {**tuning["selected"][f], "features": FEATURES_OF.get(f)} for f in ("B0", "B1", "C0")},
        "genre_entropy_threshold": _entropy_threshold(run, cfg),
        "genre_entropy_rule": "median genre entropy of the validation queries' visible histories (fixed before test)",
        "validation": fams,
        "pre_selected": pre,
        "pre_selection_rule": protocol["pre_selection"],
        "deploy_seed": deploy_seed,
        "deploy_artifact": None if pre == "A1" else next(a for a in artefacts if a["family"] == pre
                                                       and a["seed"] == deploy_seed),
        "fallback": "A1",
        "protocol_sha256": sha256_file(run.path("selection_protocol.json")),
        "inputs": {name: sha256_file(run.path(name)) for name in ("data_manifest.json", "split_manifest.json",
                                                                  "tuning.json", "models_manifest.json")},
        "note": "test users have not been evaluated by any model at freeze time",
    }
    write_json(run.path("freeze_manifest.json"), manifest)
    write_json(run.path("feature_schema.json"), feature_schema())
    mark_stage(run.root, "freeze", {"pre_selected": pre})
    log(f"[freeze] pre-selected {pre} (validation HR@8 {fams[pre]['val_HR@8']:.4f}); fallback A1 "
        f"(w={mm['a1_weight']})")
    return manifest


def _gbdt_equivalence(run, cfg, models: list[dict], n_queries: int = 40) -> dict:
    """Native LightGBM vs the NumPy reader used by the service, on real validation candidate rows."""
    import lightgbm as lgb

    from algorithm2_benchmark.pipeline.data import load_prepared
    from algorithm2_benchmark.pipeline.evaluate import r0_candidates
    from algorithm2_benchmark.pipeline.split import load_queries
    from app.recommendation.algorithm2.features import FEATURE_NAMES, compute_features
    from app.recommendation.algorithm2.gbdt_tree import TreeEnsemble
    from app.recommendation.algorithm2.profiles import build_profile

    cat = load_prepared(cfg).cat
    q = load_queries(run)
    q = q[q.partition == "val"].head(n_queries)
    F = np.concatenate([compute_features(cat, build_profile(cat, np.asarray(v, dtype=np.int64)),
                                         r0_candidates(cat, np.asarray(v, dtype=np.int64))) for v in q.visible])
    out = {"rows": int(len(F)), "queries": int(len(q)), "models": {}}
    for m in models:
        if m["family"] not in ("B0", "B1"):
            continue
        idx = [FEATURE_NAMES.index(f) for f in m["features"]]
        native = lgb.Booster(model_file=str(run.root / m["file"])).predict(F[:, idx], raw_score=True, num_threads=1)
        mine = TreeEnsemble.from_file(run.root / m["file"]).predict_raw(F[:, idx])
        out["models"][f"{m['family']}_seed{m['seed']}"] = float(np.max(np.abs(native - mine))) if len(F) else 0.0
    out["max_abs_diff"] = max(out["models"].values()) if out["models"] else 0.0
    out["equal_within_1e-9"] = out["max_abs_diff"] <= 1e-9
    return out


def _trial_rows(run) -> list[dict]:
    import pandas as pd

    p = run.path("trial_results.csv")
    return pd.read_csv(p).to_dict("records") if p.exists() else []


def decide(run, primary: dict, paired: dict) -> dict:
    """Confirmation of the pre-selected candidate on test (pre-registered rules)."""
    cfg = run.config()
    fz = read_json(run.path("freeze_manifest.json"))
    sel = cfg["selection"]
    pre = fz["pre_selected"]
    deploy_label = "A1" if pre == "A1" else f"{pre}_seed{fz['deploy_seed']}"
    cand = primary[deploy_label]
    a0 = primary["A0"]
    vs_a0 = paired.get(f"{deploy_label} vs A0", {})
    vs_a1 = paired.get(f"{deploy_label} vs A1", {})
    checks = {
        "auc_at_least": {"threshold": sel["auc_min"], "value": cand["AUC"],
                         "pass": cand["AUC"] is not None and cand["AUC"] >= sel["auc_min"]},
        "hr_gain_vs_A0": {"threshold": sel["hr_gain_min"], "value": cand["HR@8"] - a0["HR@8"],
                          "pass": cand["HR@8"] - a0["HR@8"] >= sel["hr_gain_min"]},
        "rule_violations_zero": {"value": cand["violations"], "pass": cand["violations"] == 0},
        "ci_lower_vs_A0_above_zero": {"ci95": vs_a0.get("ci95"),
                                      "pass": bool(vs_a0.get("ci95") and vs_a0["ci95"][0] is not None
                                                   and vs_a0["ci95"][0] > 0)},
    }
    if pre != "A1":
        checks["ci_lower_vs_A1_above_zero"] = {"ci95": vs_a1.get("ci95"),
                                               "pass": bool(vs_a1.get("ci95") and vs_a1["ci95"][0] is not None
                                                            and vs_a1["ci95"][0] > 0)}
    confirmed = all(c["pass"] for c in checks.values())
    if pre == "A1":
        default = "A1"
        reason = ("A1 (the conservative fallback) was pre-selected on validation; no learned model was ahead of it "
                  "by the pre-registered rule, so no learned model is deployed")
    elif confirmed:
        default = pre
        reason = f"{pre} passed every pre-registered confirmation check on test"
    else:
        default = "A1"
        failed = [k for k, c in checks.items() if not c["pass"]]
        reason = f"{pre} did not pass {', '.join(failed)}; the pre-registered fallback A1 is deployed"
    out = {"decided_at": now(), "pre_selected": pre, "deploy_label": deploy_label, "checks": checks,
           "confirmed": confirmed, "default_scorer": default,
           "default_artifact": fz["deploy_artifact"] if default == pre and pre != "A1" else None,
           "a1_hits_weight": fz["formula"]["A1"]["hits_weight"], "reason": reason}
    write_json(run.path("decision.json"), out)
    mark_stage(run.root, "decision", {"default": default})
    return out
