"""`evaluate-test` stage: the frozen models (all seeds) plus A0 / A1 and the legacy references on the
test users, rules R0-R3, strict mode. Re-running reproduces the same numbers; it never re-tunes.

Outputs: predictions.parquet (one row per query x rule x model; a working copy stays in work/), metrics.json,
summary.csv, subgroup_metrics.csv, paired_comparisons.csv, decision.json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from algorithm2_benchmark.pipeline.common import mark_stage, now, read_json, sha256_file, write_json
from algorithm2_benchmark.pipeline.data import load_prepared
from algorithm2_benchmark.pipeline.evaluate import RULE_IDS, RuleMasks, evaluate_queries, primary_mask
from algorithm2_benchmark.pipeline.metrics import K, paired_bootstrap, per_user, user_macro
from algorithm2_benchmark.pipeline.select import decide
from algorithm2_benchmark.pipeline.split import load_queries
from app.recommendation.algorithm2.bundle import load_scorers
from app.recommendation.algorithm2.profiles import build_profile

HISTORY_BUCKETS = [(1, 4, "1-4"), (5, 9, "5-9"), (10, 10**9, ">=10")]
CAND_BUCKETS = [(0, 8, "<=8"), (9, 50, "9-50"), (51, 500, "51-500"), (501, 10**9, ">500")]


def _bucket(x: int, buckets) -> str:
    for lo, hi, name in buckets:
        if lo <= x <= hi:
            return name
    return "other"


def genre_entropy(cat, history: np.ndarray) -> float:
    if len(history) == 0:
        return float("nan")
    c = np.asarray(cat.X["genre"][history].sum(axis=0)).ravel()
    if c.sum() == 0:
        return float("nan")
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def aggregate(df: pd.DataFrame) -> dict:
    ok = df["target_eligible"] & df["hit"].notna()
    d = df[ok]
    users = d["user"].to_numpy()
    out = {}
    for m, name in (("hit", "HR@8"), ("ndcg", "NDCG@8"), ("mrr", "MRR@8"), ("auc", "AUC")):
        v, n = user_macro(users, d[m].astype(float).to_numpy())
        out[name] = v
        if m == "hit":
            out["users"] = n
    out["queries"] = int(len(d))
    out["violations"] = int(df["violations"].sum())
    return out


def _checks(run) -> dict:
    if not run.frozen():
        raise SystemExit("freeze the run first (python -m algorithm2_benchmark freeze ...)")
    fz = read_json(run.path("freeze_manifest.json"))
    if sha256_file(run.path("selection_protocol.json")) != fz["protocol_sha256"]:
        raise SystemExit("selection_protocol.json changed after freeze; refusing to evaluate")
    return fz


def test_pass(run, log, which: str) -> None:
    """One isolated pass: 'gbdt' (formulas, legacy references, LightGBM native) or 'tower' (C0, PyTorch)."""
    _checks(run)
    cfg = run.config()
    prep = load_prepared(cfg)
    cat = prep.cat
    rules = RuleMasks(cat, prep.engine, cfg)
    if which == "gbdt":
        scorers, _ = load_scorers(run.root, cat, None, num_threads=int(cfg["gbdt"]["num_threads"]), include_aux=True,
                                  families=("B0", "B1"), gbdt_engine="native")
    else:
        import torch

        torch.set_num_threads(int(cfg["two_tower"]["torch_threads"]))
        loaded, _ = load_scorers(run.root, cat, None, families=("C0",))
        scorers = {k: v for k, v in loaded.items() if k.startswith("C0_")}
    test = load_queries(run)
    test = test[test.partition == "test"].reset_index(drop=True)
    log(f"[test:{which}] {len(test)} test queries x {len(RULE_IDS)} rules x {len(scorers)} scorers")
    timing: dict = {}
    t0 = time.time()
    df = evaluate_queries(cat, rules, test, scorers, RULE_IDS, K, keep_top=True, log=log, timing=timing)
    df.to_parquet(run.work / f"predictions_test_{which}.parquet", index=False)
    write_json(run.work / f"timing_test_{which}.json", {"seconds": round(time.time() - t0, 1),
                                                         "scoring_seconds": timing, "queries": int(len(test))})


def evaluate_test(run, log, spawn) -> dict:
    fz = _checks(run)
    cfg = run.config()
    spawn("test-pass-gbdt")
    spawn("test-pass-tower")
    df = pd.concat([pd.read_parquet(run.work / f"predictions_test_{w}.parquet") for w in ("gbdt", "tower")],
                   ignore_index=True)
    timing, elapsed = {}, 0.0
    for w in ("gbdt", "tower"):
        t = read_json(run.work / f"timing_test_{w}.json")
        timing.update(t["scoring_seconds"])
        elapsed += t["seconds"]
    prep = load_prepared(cfg)
    cat = prep.cat
    test = load_queries(run)
    test = test[test.partition == "test"].reset_index(drop=True)
    scorers = {m: None for m in df["model"].unique().tolist()}
    ent = {q.query_id: genre_entropy(cat, np.asarray(q.visible, dtype=np.int64)) for q in test.itertuples(index=False)}
    thr = fz.get("genre_entropy_threshold")
    df["history_bucket"] = df["history_size"].map(lambda x: _bucket(int(x), HISTORY_BUCKETS))
    df["cand_bucket"] = df["n_cand"].map(lambda x: _bucket(int(x), CAND_BUCKETS))
    df["genre_entropy"] = df["query_id"].map(ent)
    df["profile_mix"] = np.where(df["genre_entropy"].isna(), "unknown",
                                 np.where(df["genre_entropy"] <= thr, "concentrated", "mixed")) if thr is not None else "n/a"
    df.to_parquet(run.work / "predictions_test.parquet", index=False)
    export_predictions(run, df, cat)
    models = list(scorers)
    prim = df[primary_mask(df)]
    primary = {m: aggregate(prim[prim.model == m]) for m in models}
    for m in models:
        primary[m]["violations"] = int(df[df.model == m]["violations"].sum())
    fams = {}
    for fam in ("B0", "B1", "C0"):
        labels = [m for m in models if m.startswith(fam + "_seed")]
        vals = [primary[m]["HR@8"] for m in labels]
        fams[fam] = {"seeds": {m: primary[m]["HR@8"] for m in labels}, "HR@8_mean": float(np.mean(vals)),
                     "HR@8_sd": float(np.std(vals)),
                     "NDCG@8_mean": float(np.mean([primary[m]["NDCG@8"] for m in labels])),
                     "AUC_mean": float(np.mean([primary[m]["AUC"] for m in labels]))}
    # paired comparisons on the primary set (user-level; seeds averaged per user first for families)
    reps, bseed = int(cfg["selection"]["bootstrap"]), int(cfg["selection"]["bootstrap_seed"])
    per = {m: {met: per_user(prim[prim.model == m]["user"].to_numpy(), prim[prim.model == m][met].astype(float).to_numpy())
               for met in ("hit", "ndcg")} for m in models}
    for fam in ("B0", "B1", "C0"):
        labels = [m for m in models if m.startswith(fam + "_seed")]
        for met in ("hit", "ndcg"):
            us = set.intersection(*(set(per[m][met]) for m in labels))
            per[f"{fam}_mean"] = per.get(f"{fam}_mean", {})
            per[f"{fam}_mean"][met] = {u: float(np.mean([per[m][met][u] for m in labels])) for u in us}
    paired = {}
    rows = []
    for m in [*models, "B0_mean", "B1_mean", "C0_mean"]:
        for base in ("A0", "A1"):
            if m == base:
                continue
            for met, name in (("hit", "HR@8"), ("ndcg", "NDCG@8")):
                r = paired_bootstrap(per[m][met], per[base][met], reps, bseed)
                if met == "hit":
                    paired[f"{m} vs {base}"] = r
                rows.append({"model": m, "baseline": base, "metric": name, "mean_diff": r["mean_diff"],
                             "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1], "users": r["users"]})
    # the controlled comparisons of the protocol beyond the two baselines: B1-B0 (features), C0 vs B1
    dseed = int(fz["deploy_seed"])
    for m, base in (("B1_mean", "B0_mean"), ("C0_mean", "B1_mean"), ("C0_mean", "B0_mean"),
                    (f"B1_seed{dseed}", f"B0_seed{dseed}"), (f"C0_seed{dseed}", f"B1_seed{dseed}")):
        if m not in per or base not in per:
            continue
        for met, name in (("hit", "HR@8"), ("ndcg", "NDCG@8")):
            r = paired_bootstrap(per[m][met], per[base][met], reps, bseed)
            if met == "hit":
                paired[f"{m} vs {base}"] = r
            rows.append({"model": m, "baseline": base, "metric": name, "mean_diff": r["mean_diff"],
                         "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1], "users": r["users"]})
    pd.DataFrame(rows).to_csv(run.path("paired_comparisons.csv"), index=False)
    # subgroup and end-to-end tables
    sub_rows = []
    for dim in ("rule", "group", "history_bucket", "cand_bucket", "profile_mix"):
        for (m, val), g in df.groupby(["model", dim]):
            if dim != "rule":
                g = g[g.rule == "R0"]
            if g.empty:
                continue
            a = aggregate(g)
            sub_rows.append({"model": m, "dimension": dim, "value": val, **a,
                             "note": "conditional on target eligibility" if dim == "rule" else "rule R0"})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(run.path("subgroup_metrics.csv"), index=False)
    e2e, elig = {}, {}
    for (m, rule), g in df.groupby(["model", "rule"]):
        hit_any = g["hit"].fillna(0.0).astype(float)
        e2e.setdefault(m, {})[rule] = {"end_to_end_target_in_strict_top8": user_macro(g["user"].to_numpy(),
                                                                                     hit_any.to_numpy())[0],
                                       "raw_queries": int(len(g))}
        if m == "A0":
            elig[rule] = {"queries": int(len(g)), "target_eligible_share": float(g["target_eligible"].mean()),
                          "not_applicable": int((g["excluded_reason"] == "rule_not_applicable_no_genre_in_history").sum()),
                          "empty_result_share": float((g["n_cand"] == 0).mean()),
                          "fewer_than_8_share": float((g["n_cand"] < K).mean()),
                          "candidate_count_pct": {p: float(np.percentile(g["n_cand"], p)) for p in (5, 25, 50, 75, 95)},
                          "eligible_with_le8_candidates": int(((g["n_cand"] <= K) & g["target_eligible"]).sum())}
    hygiene = {m: {"rule_violation_rate": float((g["violations"] > 0).mean()),
                   "duplicate_rate": float((g["dup_in_top"] > 0).mean()),
                   "saved_in_top_rate": float((g["saved_in_top"] > 0).mean())} for m, g in df.groupby("model")}
    decision = decide(run, primary, paired)
    metrics = {"evaluated_at": now(), "level": cfg["level"], "test_queries": int(len(test)),
               "test_users": int(test.offline_user_id.nunique()), "primary_definition": "R0, main group, >8 candidates, "
               "target eligible; user-macro", "primary": primary, "families": fams, "end_to_end": e2e,
               "rule_eligibility": elig, "hygiene": hygiene, "decision": decision,
               "scoring_seconds_total": {k: round(v, 2) for k, v in timing.items()},
               "evaluation_seconds": round(elapsed, 1), "genre_entropy_threshold": thr}
    write_json(run.path("metrics.json"), metrics)
    summary = []
    for m in models:
        p = primary[m]
        fam = m.split("_seed")[0] if "_seed" in m else m
        pa0, pa1 = paired.get(f"{m} vs A0", {}), paired.get(f"{m} vs A1", {})
        summary.append({"experiment": m, "family": fam, "data": "real ratings proxy (rating>=8)" if cfg["level"] != "fixture"
                        else "synthetic/fixture", "level": cfg["level"], "users": p["users"], "queries": p["queries"],
                        "candidate_protocol": "full catalogue, strict rules", "HR@8": p["HR@8"], "AUC": p["AUC"],
                        "NDCG@8": p["NDCG@8"], "MRR@8": p["MRR@8"],
                        "HR_diff_vs_A0": pa0.get("mean_diff"), "HR_diff_vs_A0_ci95": pa0.get("ci95"),
                        "HR_diff_vs_A1": pa1.get("mean_diff"), "HR_diff_vs_A1_ci95": pa1.get("ci95"),
                        "violations": p["violations"],
                        "confirmed": (decision["confirmed"] if m == decision["deploy_label"] else None)})
    pd.DataFrame(summary).to_csv(run.path("summary.csv"), index=False)
    mark_stage(run.root, "evaluate-test", {"default": decision["default_scorer"]})
    log(f"[test] done in {elapsed:.0f}s; default scorer -> {decision['default_scorer']} ({decision['reason']})")
    return metrics


def export_predictions(run, df: pd.DataFrame, cat) -> None:
    """predictions.parquet: one row per test query x rule x model, with run / family / seed / target id
    (the catalogue's id: the MAL id for real data)."""
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    fam_seed = out["model"].str.extract(r"^(?P<family>[A-Z]\d)_seed(?P<seed>\d+)$")
    out.insert(2, "family", fam_seed["family"].fillna(out["model"]))
    out.insert(3, "seed", pd.to_numeric(fam_seed["seed"], errors="coerce").astype("Int64"))
    out.insert(out.columns.get_loc("target_key") + 1, "target_id",
               out["target_key"].map(dict(zip(cat.snap.keys.tolist(), cat.anime_id.tolist(), strict=True)))
               .astype("int64"))
    out.to_parquet(run.path("predictions.parquet"), index=False)


def entropy_threshold(cat, queries: pd.DataFrame) -> float | None:
    vals = [genre_entropy(cat, np.asarray(v, dtype=np.int64)) for v in queries.visible]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def profile_of(cat, q) -> object:
    return build_profile(cat, np.asarray(q.visible, dtype=np.int64))
