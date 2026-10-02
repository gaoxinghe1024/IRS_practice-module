"""`verify-integration` stage: the collection page serves the frozen run through the offline code path.

Runs the real web app in-process on a throwaway copy of the activated database (evaluation/sandbox.py;
the business database is never written) and records integration_report.json, one entry per check:

  catalogue_in_app_db      every catalogue member is a served title of the app database
  service_loaded           the frozen bundle loads and the default scorer is the one decision.json names
  offline_online_top8      same H, rule, catalogue and model version: the API Top-8 equals the offline test
                           prediction (A0, A1 and the deploy-seed B0 / B1 / C0)
  collection_change        adding a title to the collection changes the profile and the Top-8; removing
                           it restores the original result
  strict_short_list        filters matching fewer than 8 titles return exactly those (no padding)
  strict_no_match          no match: no items, relaxation options offered, nothing relaxed silently
  explicit_relaxation      applying an option: relaxed=true, the relaxed rule, the strict result count
  detail_pages             every recommended id opens its API detail and its web page
  missing_bundle_fallback  an app configured with a missing run serves the formula, reports degraded and
                           still applies the user's filters strictly
  incompatible_bundle      a changed model file or catalogue hash is refused (formula fallback, recorded)

Real data only; the fixture level records not_applicable.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import PROJECT_ROOT
from app.core.text import norm
from app.recommendation.constraints import ConstraintSet
from evaluation.algorithm2.common import mark_stage, now, read_json, write_json
from evaluation.algorithm2.data import load_prepared
from evaluation.algorithm2.split import load_queries
from evaluation.algorithm2.webapp import Account, rule_body, served_app

CORE = ("A0", "A1", "B0", "B1", "C0")
RULES = ("R0", "R1", "R2", "R3")


def _check(name: str, passed: bool, **details) -> dict:
    return {"check": name, "passed": bool(passed), **details}


def _keys(resp: dict) -> list[str]:
    return [it["item_key"] for it in resp.get("items", [])]


def _agreement(run, cfg, cat, acct, fz, log, n_main: int, n_sparse: int) -> tuple[dict, dict]:
    """API Top-8 vs offline predictions for the same queries, rules and models."""
    pred = pd.read_parquet(run.path("predictions.parquet"),
                           columns=["query_id", "group", "rule", "model", "top_keys", "excluded_reason"])
    q = load_queries(run)
    test = q[q.partition == "test"].sort_values("query_id")
    chosen = pd.concat([test[test.group == "main"].head(n_main), test[test.group == "sparse"].head(n_sparse)])
    seed = int(fz["deploy_seed"])
    label = {m: (m if m in ("A0", "A1") else f"{m}_seed{seed}") for m in CORE}
    offline = {(r.query_id, r.rule, r.model): r.top_keys for r in pred.itertuples(index=False)}
    r2 = cfg["rules"]["R2"]
    cids, compared, mismatches, skipped = {}, 0, [], 0
    for qrow in chosen.itertuples(index=False):
        cids[qrow.query_id] = acct.collection(f"agree {len(cids) + 1}",
                                              cat.snap.keys[np.asarray(qrow.visible)].tolist())
        for rule in RULES:
            body = rule_body(rule, qrow.r1_keyword, r2)
            if body is None:
                skipped += 1
                continue
            for m in CORE:
                resp = acct.recommend(cids[qrow.query_id], {**body, "scorer": m})
                if resp.status_code != 200:
                    mismatches.append({"query_id": qrow.query_id, "rule": rule, "model": m,
                                       "error": f"HTTP {resp.status_code}"})
                    continue
                api = _keys(resp.json())
                off = offline.get((qrow.query_id, rule, label[m]))
                off = off.split("|") if off else []
                compared += 1
                if api != off:
                    mismatches.append({"query_id": qrow.query_id, "rule": rule, "model": m, "api": api,
                                       "offline": off})
    log(f"[integration] offline/online Top-8: {compared - len(mismatches)}/{compared} identical")
    check = _check("offline_online_top8", compared > 0 and not mismatches, compared=compared,
                   identical=compared - len(mismatches), mismatches=mismatches[:20],
                   queries={"main": int((chosen.group == "main").sum()), "sparse": int((chosen.group == "sparse").sum())},
                   rule_requests_without_keyword_skipped=skipped,
                   offline_labels=label, note="offline B0/B1 rows were scored by native LightGBM, the service uses "
                                              "the NumPy reader of the same model files")
    return check, {"cids": cids, "chosen": chosen}


def _collection_change(acct, cat, chosen) -> dict:
    """Small collection + one title with different genres: profile and Top-8 change; removal restores."""
    sparse = chosen[chosen.group == "sparse"]
    qrow = (sparse if len(sparse) else chosen).iloc[0]
    H = np.asarray(qrow.visible, dtype=np.int64)
    cid = acct.collection("change check", cat.snap.keys[H].tolist())
    before = acct.recommend(cid, {}).json()
    h_genres = set(np.flatnonzero(np.asarray(cat.X["genre"][H].sum(axis=0)).ravel()).tolist())
    top = set(_keys(before))
    members = cat.members[np.argsort(-np.nan_to_num(cat.snap.members[cat.members]), kind="stable")]
    add = None
    for i in members.tolist():
        g = set(cat.X["genre"][i].indices.tolist())
        if g and not g & h_genres and i not in set(H.tolist()) and str(cat.snap.keys[i]) not in top:
            add = i
            break
    if add is None:
        return _check("collection_change", False, reason="no title with disjoint genres found")
    key = str(cat.snap.keys[add])

    def results() -> dict:
        return {m: acct.recommend(cid, {"scorer": m}).json() for m in CORE}

    res_before = results()
    acct.add(cid, key)
    after = acct.recommend(cid, {}).json()
    res_after = results()
    acct.remove(cid, key)
    res_restored = results()
    changed = {m: _keys(res_after[m]) != _keys(res_before[m]) for m in CORE}
    restored_ok = all(_keys(res_restored[m]) == _keys(res_before[m]) and
                      [it["score"] for it in res_restored[m]["items"]] == [it["score"] for it in res_before[m]["items"]]
                      for m in CORE)
    profile_changed = after["profile"]["top_tags"] != before["profile"]["top_tags"]
    return _check("collection_change", profile_changed and any(changed.values()) and restored_ok
                  and all(key not in _keys(res_after[m]) for m in CORE),
                  history_size=int(len(H)), added=key, profile_before=before["profile"]["top_tags"],
                  profile_after=after["profile"]["top_tags"], top8_changed_by_scorer=changed,
                  default_scorer=before["meta"]["algorithm"],
                  default_top8_changed=_keys(after) != _keys(before), removal_restores_every_scorer=restored_ok,
                  note="the profile is rebuilt from the current collection on every request; whether the Top-8 "
                       "moves depends on how strongly the scorer weighs the profile")


def _short_list(acct, svc, cid, keys: list[str]) -> dict:
    """A title word that leaves 1-7 eligible titles for this collection: exactly those come back."""
    cat = svc.cat
    H = np.asarray([cat.snap.index[k] for k in keys], dtype=np.int64)
    for i in cat.members.tolist():
        words = [w for w in norm(str(cat.snap.titles[i])).split() if len(w) >= 7 and w.isalpha()]
        for w in words:
            n = svc.candidates(H, ConstraintSet(keywords=[w]))[1].count
            if 1 <= n <= 7:
                resp = acct.recommend(cid, {"keywords": w}).json()
                items = resp["items"]
                ok = (resp["status"] == "ok" and len(items) == n and not resp["relaxed"]
                      and not resp["relaxation_options"]
                      and any(f"Only {n} titles" in x for x in resp["meta"]["warnings"]))
                return _check("strict_short_list", ok, keyword=w, eligible=n, returned=len(items),
                              warnings=resp["meta"]["warnings"])
    return _check("strict_short_list", False, reason="no keyword with 1-7 matches found")


def _no_match_and_relaxation(acct, cid) -> list[dict]:
    out = []
    kw = acct.recommend(cid, {"keywords": "zzqx no such title"}).json()
    out.append(_check("strict_no_match_keyword", kw["status"] == "no_match" and kw["items"] == [] and not kw["relaxed"]
                      and kw["constraints_satisfied"], status=kw["status"], options=kw["relaxation_options"]))
    strict_body = {"max_episodes": 1, "min_score": 9.9}
    strict = acct.recommend(cid, strict_body).json()
    ok = (strict["status"] == "no_match" and strict["items"] == [] and not strict["relaxed"]
          and strict["relaxation_options"] and strict["strict_result_count"] is None)
    out.append(_check("strict_no_match", ok, status=strict["status"], options=strict["relaxation_options"],
                      warnings=strict["meta"]["warnings"]))
    if not strict["relaxation_options"]:
        out.append(_check("explicit_relaxation", False, reason="no relaxation option offered"))
        return out
    opt = strict["relaxation_options"][0]
    body = {**strict_body, **{k: v for k, v in opt["change"].items() if k in ("max_episodes", "min_score")},
            "relaxation_applied": {"rule_id": opt["rule_id"], "change": opt["change"], "original": opt.get("original")}}
    rel = acct.recommend(cid, body).json()
    eps_cap, min_score = body.get("max_episodes"), body.get("min_score")
    within = all((eps_cap is None or (it["episodes"] is not None and it["episodes"] <= eps_cap))
                 and (min_score is None or (it["catalog_score"] is not None and it["catalog_score"] >= min_score))
                 for it in rel["items"])
    ok = (rel["status"] == "ok" and rel["relaxed"] is True and rel["relaxed_rules"] == [opt["rule_id"]]
          and rel["strict_result_count"] == 0 and 0 < len(rel["items"]) <= 8 and within)
    out.append(_check("explicit_relaxation", ok, option=opt, relaxed=rel["relaxed"], relaxed_rules=rel["relaxed_rules"],
                      strict_result_count=rel["strict_result_count"], returned=len(rel["items"]),
                      items_within_relaxed_filters=within))
    return out


def _detail_pages(client, keys: set[str]) -> dict:
    bad = []
    for k in sorted(keys):
        a = client.get(f"/api/v1/anime/{k}")
        p = client.get(f"/anime/{k}")
        if a.status_code != 200 or p.status_code != 200 or a.json().get("item_key", k) != k:
            bad.append({"item_key": k, "api": a.status_code, "page": p.status_code})
    return _check("detail_pages", not bad and bool(keys), checked=len(keys), failures=bad[:20])


def _incompatible(run, svc_app, decision) -> list[dict]:
    from app.recommendation.algorithm2.service import build_service

    fz = read_json(run.path("freeze_manifest.json"))
    seed = int(fz["deploy_seed"])
    out = []
    for case in ("model_file_changed", "catalogue_hash_changed"):
        tmp = Path(tempfile.mkdtemp(prefix="alg2-bundle-"))
        try:
            for name in ("freeze_manifest.json", "decision.json"):
                shutil.copy(run.path(name), tmp / name)
            shutil.copytree(run.root / "models", tmp / "models")
            if case == "model_file_changed":
                art = next(a for a in fz["models"] if a["family"] == "B1" and int(a["seed"]) == seed)
                p = tmp / art["file"]
                p.write_bytes(p.read_bytes() + b"\n")
            else:
                f = json.loads((tmp / "freeze_manifest.json").read_text())
                f["catalog_hash"] = "0" * 64
                (tmp / "freeze_manifest.json").write_text(json.dumps(f))
            cc = {"scorer": decision["default_scorer"], "algorithm2_run": str(tmp)}
            svc = build_service(svc_app.cat.snap, svc_app.engine, cc, PROJECT_ROOT)
            ok = (svc.status == "degraded" and svc.default == "A0"
                  and not any(m in svc.scorers for m in ("A1", "B0", "B1", "C0")))
            out.append(_check(f"incompatible_bundle:{case}", ok, status=svc.status, default=svc.default,
                              loaded=sorted(svc.scorers), detail=svc.detail[:300]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def _missing_bundle(decision, r2, H_keys, keyword) -> list[dict]:
    out = []
    missing = str(Path(tempfile.gettempdir()) / "alg2-no-such-run")
    with served_app(missing, decision["default_scorer"]) as (app, client):
        st = app.state.holder.get()
        svc, snap = st.algorithm2, st.snap
        cap = st.cap("collection_scorer")
        acct = Account(client, "alg2_fallback")
        cid = acct.collection("fallback check", H_keys)
        body = rule_body("R3", keyword, r2) or rule_body("R2", None, r2)
        resp = acct.recommend(cid, body).json()
        kw = body.get("keywords")
        idx = [snap.index[k] for k in _keys(resp)]
        strict_ok = all(snap.score[i] >= body["min_score"] and np.isfinite(snap.episodes[i])
                        and snap.episodes[i] <= body["max_episodes"] for i in idx)
        if kw:
            strict_ok = strict_ok and all(st.constraints.keyword_mask(kw)[i] for i in idx)
        out.append(_check("missing_bundle_fallback", svc is not None and svc.status == "degraded"
                          and svc.default in ("A0", "A1") and cap.status == "degraded"
                          and resp["meta"]["scorer_status"] == "degraded" and not resp["relaxed"] and strict_ok
                          and all("real-" not in w and "Error" not in w for w in resp["meta"]["warnings"]),
                          service_status=svc.status if svc else None, default=svc.default if svc else None,
                          capability=cap.status, request=body, returned=len(idx), strict_filters_held=strict_ok,
                          user_warnings=resp["meta"]["warnings"]))
        nm = acct.recommend(cid, {"max_episodes": 1, "min_score": 9.9}).json()
        out.append(_check("missing_bundle_no_silent_relaxation", nm["status"] == "no_match" and nm["items"] == []
                          and not nm["relaxed"], status=nm["status"]))
    return out


def verify_integration(run, log) -> dict:
    cfg = run.config()
    started = time.time()
    if cfg["level"] == "fixture":
        rep = {"verified_at": now(), "level": "fixture", "status": "not_applicable",
               "reason": "fixture data is not served by the web app; run verify-integration on a smoke/main run"}
        write_json(run.path("integration_report.json"), rep)
        mark_stage(run.root, "verify-integration", {"status": "not_applicable"})
        return rep
    prep = load_prepared(cfg)
    cat = prep.cat
    fz = read_json(run.path("freeze_manifest.json"))
    decision = read_json(run.path("decision.json"))
    checks: list[dict] = []
    with served_app(str(run.root), decision["default_scorer"]) as (app, client):
        st = app.state.holder.get()
        svc = st.algorithm2
        # 1. the catalogue the service needs is in the app database (non-destructive check on the copy)
        db_path = app.state.settings.db_path
        with sqlite3.connect(db_path) as con:
            served = {r[0] for r in con.execute("SELECT item_key FROM anime WHERE in_catalog = 1 AND is_explicit = 0")}
        member_keys = set(cat.snap.keys[cat.members].tolist())
        missing = sorted(member_keys - served)
        checks.append(_check("catalogue_in_app_db", not missing, catalogue_members=len(member_keys),
                             missing=len(missing), examples=missing[:10], database="throwaway copy of the activated "
                                                                                    "real database"))
        checks.append(_check("service_loaded", svc is not None and svc.status == "ready"
                             and svc.default == decision["default_scorer"]
                             and svc.cat.catalog_hash == fz["catalog_hash"] and all(m in svc.scorers for m in CORE),
                             status=svc.status if svc else None, default=svc.default if svc else None,
                             expected_default=decision["default_scorer"],
                             catalog_hash=svc.cat.catalog_hash[:12] if svc else None,
                             scorers=sorted(svc.scorers) if svc else [],
                             capability=st.cap("collection_scorer").status))
        if svc is None or svc.status != "ready":
            rep = {"verified_at": now(), "level": cfg["level"], "status": "failed", "checks": checks}
            write_json(run.path("integration_report.json"), rep)
            raise SystemExit("the sandbox app did not load the frozen run; see integration_report.json")
        acct = Account(client, "alg2_integration")
        icfg = cfg.get("integration", {})
        agree, ctx = _agreement(run, cfg, cat, acct, fz, log, int(icfg.get("main_queries", 40)),
                                int(icfg.get("sparse_queries", 10)))
        checks.append(agree)
        checks.append(_collection_change(acct, cat, ctx["chosen"]))
        q0 = ctx["chosen"].iloc[0]
        H0 = np.asarray(q0.visible, dtype=np.int64)
        cid0 = ctx["cids"][q0.query_id]
        checks.append(_short_list(acct, svc, cid0, cat.snap.keys[H0].tolist()))
        checks += _no_match_and_relaxation(acct, cid0)
        rec_keys: set[str] = set()
        for cid in list(ctx["cids"].values())[:10]:
            for body in ({}, {"min_score": 7.0, "max_episodes": 26}):
                rec_keys |= set(_keys(acct.recommend(cid, body).json()))
        checks.append(_detail_pages(client, rec_keys))
        checks.append(_check("collection_page_renders", client.get(f"/collections/{cid0}").status_code == 200))
        checks += _incompatible(run, svc, decision)
        H_keys = cat.snap.keys[H0].tolist()
    checks += _missing_bundle(decision, cfg["rules"]["R2"], H_keys, q0.r1_keyword)
    passed = all(c["passed"] for c in checks)
    rep = {"verified_at": now(), "level": cfg["level"], "run_id": run.run_id,
           "status": "passed" if passed else "failed", "default_scorer": decision["default_scorer"],
           "database": "throwaway copy of the activated real database (evaluation/sandbox.py); the business "
                       "database was not modified",
           "seconds": round(time.time() - started, 1), "checks": checks}
    write_json(run.path("integration_report.json"), rep)
    mark_stage(run.root, "verify-integration", {"status": rep["status"]})
    log(f"[integration] {sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    if not passed:
        raise SystemExit("integration checks failed: " + ", ".join(c["check"] for c in checks if not c["passed"]))
    return rep
