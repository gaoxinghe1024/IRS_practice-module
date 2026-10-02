"""`benchmark` stage: latency and cost of the frozen scorers as they are served.

Fixed for every model: this machine, `latency.threads` CPU threads, the run's catalogue and one request
list (main-group test queries in query_id order, up to `latency.requests`; the warm-up requests are
main-group validation queries and are not measured). Each request is timed at three layers:

  scoring  the scorer alone on precomputed inputs (the eight features for A/B, the profile for C0)
  service  Alg2Service.recommend with H given: profile, strict rules, features, scoring, Top-8
  api      POST /api/v1/collections/{id}/recommendations in-process on a throwaway copy of the app
           database: reads H from the database, everything above, reasons, run log (real data only)

for rules R0 and R3. B0/B1 run on the NumPy reader of the LightGBM text model and C0 on PyTorch (CPU),
exactly as served. No result cache, no batching, no GPU. Loading (catalogue, models, item embeddings)
is timed separately and never mixed into request latency.
Outputs: latency.csv, benchmark.json.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.recommendation.algorithm2.bundle import load_scorers
from app.recommendation.algorithm2.features import compute_features
from app.recommendation.algorithm2.profiles import build_profile
from app.recommendation.algorithm2.scorers import QueryContext
from app.recommendation.algorithm2.service import Alg2Service
from evaluation.algorithm2.common import (
    environment,
    mark_stage,
    now,
    peak_rss_mb,
    read_json,
    sha256_bytes,
    write_json,
)
from evaluation.algorithm2.data import load_prepared
from evaluation.algorithm2.evaluate import RuleMasks
from evaluation.algorithm2.split import load_queries

BENCH_RULES = ("R0", "R3")
MODELS = ("A0", "A1", "B0", "B1", "C0")
ENGINE = {"A0": "formula (NumPy)", "A1": "formula (NumPy)", "B0": "LightGBM text model, NumPy reader",
          "B1": "LightGBM text model, NumPy reader", "C0": "PyTorch CPU (item vectors precomputed at load)"}


def _stats(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "max_ms": None}
    a = np.asarray(ms, dtype=float)
    return {"n": int(len(a)), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "mean_ms": round(float(a.mean()), 3),
            "max_ms": round(float(a.max()), 3)}


def request_lists(run, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = load_queries(run)
    lat = cfg["latency"]
    main = q[q.group == "main"]
    measured = main[main.partition == "test"].sort_values("query_id").head(int(lat["requests"]))
    warm = main[main.partition == "val"].sort_values("query_id").head(int(lat["warmup"]))
    return warm.reset_index(drop=True), measured.reset_index(drop=True)


def _load(run, cat, log) -> tuple[dict, dict, dict]:
    """The served objects (deploy-seed models, NumPy GBDT engine), with load times per family."""
    fz = read_json(run.path("freeze_manifest.json"))
    seed = int(fz["deploy_seed"])
    load_s, scorers = {}, {}
    t0 = time.perf_counter()
    base, _ = load_scorers(run.root, cat, [], 1, families=())
    load_s["A0+A1"] = time.perf_counter() - t0
    scorers.update(base)
    for fam in ("B0", "B1", "C0"):
        t0 = time.perf_counter()
        got, _ = load_scorers(run.root, cat, [f"{fam}_seed{seed}"], 1, families=(fam,))
        load_s[fam] = time.perf_counter() - t0
        scorers[fam] = got[f"{fam}_seed{seed}"]
    # C0 load split into reading the weights, encoding the catalogue and precomputing item vectors
    from app.recommendation.algorithm2.two_tower import TowerEncoder, item_embeddings, load_model

    art = next(a for a in fz["models"] if a["family"] == "C0" and int(a["seed"]) == seed)
    t0 = time.perf_counter()
    model, blob = load_model(run.root / art["file"])
    t1 = time.perf_counter()
    enc = TowerEncoder(cat, int(blob["encoder"]["studio_min_count"]))
    mat = enc.item_matrix()
    t2 = time.perf_counter()
    item_embeddings(model, mat)
    t3 = time.perf_counter()
    c0 = {"read_weights_s": round(t1 - t0, 4), "encode_catalogue_s": round(t2 - t1, 4),
          "item_vectors_s": round(t3 - t2, 4), "items": int(mat.shape[0])}
    files = []
    mm = {(m["family"], int(m["seed"])): m for m in read_json(run.path("models_manifest.json"))["models"]}
    for a in fz["models"]:
        info = mm.get((a["family"], int(a["seed"])), {})
        files.append({"label": f"{a['family']}_seed{a['seed']}", "file": a["file"],
                      "bytes": int((run.root / a["file"]).stat().st_size), "fit_seconds": info.get("fit_seconds"),
                      "deployed_artifact": int(a["seed"]) == seed})
    log(f"[bench] loaded {list(scorers)}: " + ", ".join(f"{k} {v:.2f}s" for k, v in load_s.items()))
    return scorers, {k: round(v, 4) for k, v in load_s.items()} | {"C0_breakdown": c0}, {"files": files,
                                                                                       "deploy_seed": seed}


def _inprocess(cat, engine, rules, scorers, warm, measured, k, log) -> tuple[list, dict]:
    svc = Alg2Service(cat, engine, scorers, "A0", "ready", "benchmark")
    rows, counts = [], {}
    for rule in BENCH_RULES:
        feat_ms: list[float] = []
        for model in MODELS:
            sc = scorers[model]
            score_ms, service_ms = [], []
            for phase, lst in (("warmup", warm), ("measure", measured)):
                for q in lst.itertuples(index=False):
                    cs = rules.constraint_set(rule, q.r1_keyword)
                    if cs is None:
                        continue
                    H = np.asarray(q.visible, dtype=np.int64)
                    t0 = time.perf_counter()
                    svc.recommend(H, cs, model, k)
                    t1 = time.perf_counter()
                    prof = build_profile(cat, H)
                    cand, _ = svc.candidates(prof.history, cs)
                    ctx = QueryContext(cat, prof, cand)
                    if model != "C0":
                        tf = time.perf_counter()
                        ctx._features = compute_features(cat, prof, cand)
                        tf1 = time.perf_counter()
                    ts = time.perf_counter()
                    if len(cand):
                        sc.score(ctx)
                    ts1 = time.perf_counter()
                    if phase == "measure":
                        service_ms.append((t1 - t0) * 1000)
                        score_ms.append((ts1 - ts) * 1000)
                        if model == "A0":
                            feat_ms.append((tf1 - tf) * 1000)
            counts[(rule, model)] = len(service_ms)
            rows.append({"layer": "scoring", "model": model, "rule": rule, **_stats(score_ms)})
            rows.append({"layer": "service", "model": model, "rule": rule, **_stats(service_ms)})
            log(f"[bench] {rule} {model}: scoring p50 {rows[-2]['p50_ms']} ms, service p50 {rows[-1]['p50_ms']} ms "
                f"p95 {rows[-1]['p95_ms']} ms (n={rows[-1]['n']})")
        rows.append({"layer": "features", "model": "A0-B1 (shared eight features)", "rule": rule, **_stats(feat_ms)})
    return rows, counts


def _api(run, cfg, cat, warm, measured, log) -> tuple[list, dict]:
    from evaluation.algorithm2.webapp import Account, rule_body, served_app

    r2 = cfg["rules"]["R2"]
    decision = read_json(run.path("decision.json"))
    info: dict = {}
    rows = []
    t0 = time.perf_counter()
    with served_app(str(run.root), decision["default_scorer"]) as (app, client):
        info["app_startup_s"] = round(time.perf_counter() - t0, 2)
        svc = app.state.holder.get().algorithm2
        info["service_status"] = svc.public_status() if svc else None
        if svc is None or svc.status != "ready" or not all(m in svc.scorers for m in MODELS):
            info["skipped"] = "the sandbox app did not load the frozen run (see service_status)"
            return rows, info
        info["catalog_hash_match"] = svc.cat.catalog_hash == cat.catalog_hash
        acct = Account(client, "alg2_benchmark")
        t0 = time.perf_counter()
        cids = {}
        for q in pd.concat([warm, measured]).itertuples(index=False):
            cids[q.query_id] = acct.collection(f"bench {len(cids) + 1}", cat.snap.keys[np.asarray(q.visible)].tolist())
        info["setup_collections_s"] = round(time.perf_counter() - t0, 1)
        info["collections"] = len(cids)
        for rule in BENCH_RULES:
            for model in MODELS:
                ms = []
                for phase, lst in (("warmup", warm), ("measure", measured)):
                    for q in lst.itertuples(index=False):
                        body = rule_body(rule, q.r1_keyword, r2)
                        if body is None:
                            continue
                        body["scorer"] = model
                        t1 = time.perf_counter()
                        r = acct.recommend(cids[q.query_id], body)
                        dt = (time.perf_counter() - t1) * 1000
                        if r.status_code != 200:
                            raise RuntimeError(f"API {rule} {model}: HTTP {r.status_code} {r.text[:200]}")
                        if r.json()["meta"].get("fallback"):
                            raise RuntimeError(f"API {rule} {model}: scorer fell back ({r.json()['meta']['fallback']})")
                        if phase == "measure":
                            ms.append(dt)
                rows.append({"layer": "api", "model": model, "rule": rule, **_stats(ms)})
                log(f"[bench] api {rule} {model}: p50 {rows[-1]['p50_ms']} ms p95 {rows[-1]['p95_ms']} ms "
                    f"(n={rows[-1]['n']})")
    return rows, info


def benchmark(run, log) -> dict:
    cfg = run.config()
    lat = cfg["latency"]
    threads = int(lat.get("threads", 1))
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    prep = load_prepared(cfg)
    cat_load_s = time.perf_counter() - t0
    rss = {"after_catalogue": peak_rss_mb()}
    rules = RuleMasks(prep.cat, prep.engine, cfg)
    warm, measured = request_lists(run, cfg)
    list_hash = sha256_bytes(",".join(measured.query_id.astype(str)).encode())[:16]
    log(f"[bench] {len(warm)} warm-up + {len(measured)} measured requests, rules {BENCH_RULES}, threads={threads}")
    with threadpool_limits(limits=threads):
        scorers, load, files = _load(run, prep.cat, log)
        import torch

        torch.set_num_threads(threads)
        rss["after_models"] = peak_rss_mb()
        rows, counts = _inprocess(prep.cat, prep.engine, rules, scorers, warm, measured, int(cfg["k"]), log)
        rss["after_inprocess"] = peak_rss_mb()
        api_info: dict = {}
        if prep.source["kind"] == "real":
            api_rows, api_info = _api(run, cfg, prep.cat, warm, measured, log)
            rows += api_rows
            rss["after_api"] = peak_rss_mb()
        else:
            api_info = {"skipped": "not_applicable: fixture data is not served by the web app"}
    fit = {f["label"]: f for f in files["files"]}
    for r in rows:
        r.update({"run_id": run.run_id, "level": cfg["level"], "threads": threads, "device": "cpu",
                  "engine": ENGINE.get(r["model"], "NumPy"), "cache": "none (rule masks per keyword are cached)",
                  "batching": "none (one request at a time)", "n_warmup": int(len(warm)),
                  "request_list_sha": list_hash})
        lab = f"{r['model']}_seed{files['deploy_seed']}"
        r["model_bytes"] = fit[lab]["bytes"] if lab in fit else 0
        r["fit_seconds"] = fit[lab]["fit_seconds"] if lab in fit else 0.0
    cols = ["run_id", "level", "layer", "model", "rule", "n_warmup", "n", "p50_ms", "p95_ms", "mean_ms", "max_ms",
            "threads", "device", "engine", "cache", "batching", "model_bytes", "fit_seconds", "request_list_sha"]
    pd.DataFrame(rows)[cols].to_csv(run.path("latency.csv"), index=False)
    out = {"measured_at": now(), "environment": environment(), "threads": threads, "device": "cpu", "gpu": False,
           "request_list": {"measured": "main-group test queries in query_id order", "n": int(len(measured)),
                            "warmup": "main-group validation queries in query_id order", "warmup_n": int(len(warm)),
                            "rules": list(BENCH_RULES), "query_ids_sha": list_hash,
                            "R3_requests_without_keyword_skipped": int(measured.r1_keyword.isna().sum())},
           "load_seconds": {"catalogue_and_rule_engine": round(cat_load_s, 2), **load}, "model_files": files,
           "api": api_info, "peak_rss_mb": rss,
           "notes": ["P50/P95 are single-request wall-clock times on this machine; they are observations, not an SLO "
                     "(the project has no agreed latency SLO).",
                     "scoring = scorer only; service = profile + strict rules + features + scoring + Top-8 with H given; "
                     "api = the collection endpoint in-process (database read of H, reasons, run log, JSON).",
                     "peak RSS is the whole benchmark process (catalogue, all five scorers and, for api, the full app).",
                     "B0/B1 are served by the NumPy reader of the LightGBM text model (no native LightGBM next to "
                     "PyTorch); freeze_manifest.json records the equivalence check against native LightGBM."]}
    write_json(run.path("benchmark.json"), out)
    mark_stage(run.root, "benchmark", {"rows": len(rows)})
    return out


def latency_table(run) -> pd.DataFrame | None:
    p = Path(run.path("latency.csv"))
    return pd.read_csv(p) if p.exists() else None
