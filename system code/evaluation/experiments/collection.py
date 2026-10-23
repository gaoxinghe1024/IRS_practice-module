"""Collection recommendations (algorithm 2): scorer comparison under one fixed rule layer.

Cases are leave-one-out: the hidden title never enters the profile; positives and negatives come from
the same eligible pool. Sources:
  collection_events      real collection events (data/evaluation/collection_events.jsonl) when present,
                         otherwise the synthetic generator's independent collection process
  proxy_liked_history    REAL mode without collections: a user's train titles rated >= 8 play the role
                         of a collection and their held-out test positive is the hidden title. This is a
                         proxy benchmark, not a collection task.
Metrics: Hits@8, NDCG@8, AUC against the whole eligible pool and AUC against 50 sampled unobserved
titles (these negatives are not proven dislikes). Scorers are compared only on cases whose hidden title
passes the rule layer (the share of such cases is reported).
"""

from __future__ import annotations

import numpy as np

from app.recommendation.collection.cases import LOOCase, build_cases, sample_negatives, split_owner
from app.recommendation.collection.gradient_boosting import (
    GBDTCollectionScorer,
    collection_features,
    fit_gbdt,
)
from app.recommendation.collection.manual import CollectionProfileScorer, ManualSlidesScorer
from app.recommendation.profiles import build_collection_profile
from evaluation.common import Run, md_table
from evaluation.metrics import auc_pos_vs_neg, bootstrap_ci, hits_at_k, ndcg_binary, paired_bootstrap_diff
from evaluation.protocols import load_context, sample_users
from pipelines.common import read_jsonl
from pipelines.fit import collection_event_source
from pipelines.split import load_split

K = 8


def _proxy_cases(ctx, users, seed) -> list[LOOCase]:
    test = load_split(ctx.settings, "ranking", "test", ctx.version)
    val = load_split(ctx.settings, "ranking", "val", ctx.version)
    tpos = dict(zip(test.offline_user_id.astype(int), test.item_idx.astype(int), strict=True))
    vpos = dict(zip(val.offline_user_id.astype(int), val.item_idx.astype(int), strict=True))
    hists = ctx.histories(users)
    cases = []
    for u in users:
        if u not in hists or u not in tpos:
            continue
        items, r = hists[u]
        liked = [int(i) for i, x in zip(items, r, strict=True) if x >= 8]
        if len(liked) < 3:
            continue
        m = ctx.pool_mask.copy()
        m[items] = False
        if u in vpos:
            m[vpos[u]] = False
        prof = build_collection_profile(ctx.snap, liked)
        cases.append(LOOCase(f"proxy:{u}", str(u), f"proxy:{u}", liked, tpos[u], prof, np.flatnonzero(m), bool(m[tpos[u]])))
    return cases


def _evaluate(ctx, cases, scorers, run, source, rng) -> tuple[dict, list[dict]]:
    per = {name: [] for name in scorers}
    for c in cases:
        if not c.target_eligible:
            continue
        neg50 = sample_negatives(c, 50, rng)
        for name, sc in scorers.items():
            s, _ = sc.score(ctx.snap, c.profile, c.eligible)
            s = np.asarray(s, dtype=np.float64)
            order = np.lexsort((ctx.snap.keys[c.eligible], -s))
            ranked = c.eligible[order][:K].tolist()
            pos = np.flatnonzero(c.eligible == c.target)[0]
            others = np.delete(s, pos)
            neg_idx = np.searchsorted(c.eligible, neg50)
            m = {"hits@8": hits_at_k(ranked, {c.target}, K), "ndcg@8": ndcg_binary(ranked, {c.target}, K),
                 "auc_full_pool": auc_pos_vs_neg([s[pos]], others),
                 "auc_sampled50": auc_pos_vs_neg([s[pos]], s[neg_idx])}
            per[name].append(m)
            run.case({"source": source, "case": c.case_id, "scorer": name, **{k: round(v, 5) for k, v in m.items()}})
    agg = {name: {k: bootstrap_ci([m[k] for m in v]) for k in ("hits@8", "ndcg@8", "auc_full_pool", "auc_sampled50")}
           for name, v in per.items() if v}
    return agg, per


def run(settings, sample_users_n: int | None = None, tag=None, **_) -> dict:
    ctx = load_context(settings, load_models=False)
    seed = int(settings.pipeline.get("split_seed", 20260930))
    run = Run(settings, "collection", {"k": K, "sample_users": sample_users_n, "tag": tag})
    rng = np.random.default_rng(3)
    base_scorers = {"manual_slides_v0": ManualSlidesScorer(),
                    "collection_profile_v1": CollectionProfileScorer(float(settings.recommendation["collection"].get("alpha", 0.7)))}
    metrics, rows, decisions = {"sources": {}}, [], []
    src = collection_event_source(settings)
    sources = []
    if src is not None:
        events = list(read_jsonl(src[1]))
        cases = build_cases(ctx.snap, events, seed)
        parts = {c.case_id: split_owner(c.owner, seed) for c in cases}
        scorers = dict(base_scorers)
        gb = ctx.manifest.get("components", {}).get("gbdt", {})
        if gb.get("status") == "trained":
            import joblib

            scorers["gradient_boosting_v1"] = GBDTCollectionScorer(joblib.load(ctx.art / "models" / "gbdt_collection.joblib"), gb)
        tt = ctx.manifest.get("components", {}).get("two_tower", {})
        if tt.get("status") == "trained":
            try:
                from app.recommendation.collection.two_tower import TwoTowerScorer

                scorers["two_tower_v1"] = TwoTowerScorer.load(ctx.art / "models" / "two_tower.pt", ctx.snap)
            except Exception as exc:  # noqa: BLE001
                run.notes.append(f"two-tower not loaded: {exc}")
        for split in ("val", "test"):
            sc = [c for c in cases if parts[c.case_id] == split]
            sources.append((f"{src[0]}:{split}", sc, scorers))
    else:
        run.notes.append("No real collection events: collection scorers are compared on the proxy_liked_history "
                         "benchmark (NOT a collection task).")
        test = load_split(settings, "ranking", "test", ctx.version)
        n = sample_users_n or 1500
        users = sample_users(test, n * 3, seed=31)
        train_users, val_users, eval_users = users[n:2 * n], users[2 * n:], users[:n]
        tr_cases = _proxy_cases(ctx, train_users, seed)
        va_cases = _proxy_cases(ctx, val_users, seed)
        scorers = dict(base_scorers)

        def rows_of(cs):
            X, y = [], []
            for c in cs:
                if not c.target_eligible:
                    continue
                neg = sample_negatives(c, 4, rng)
                cand = np.concatenate([[c.target], neg])
                X.append(collection_features(ctx.snap, c.profile, cand))
                y.append(np.r_[1, np.zeros(len(neg))])
            return np.vstack(X), np.concatenate(y)
        if tr_cases and va_cases:
            model, meta = fit_gbdt(*rows_of(tr_cases), *rows_of(va_cases), seed=0)
            scorers["gradient_boosting_v1_proxy"] = GBDTCollectionScorer(model, meta)
            metrics["gbdt_proxy_training"] = meta
        sources.append(("proxy_liked_history:val", va_cases, scorers))
        sources.append(("proxy_liked_history:test", _proxy_cases(ctx, eval_users, seed), scorers))
    for name, cases, scorers in sources:
        eligible_share = float(np.mean([c.target_eligible for c in cases])) if cases else float("nan")
        agg, per = _evaluate(ctx, cases, scorers, run, name, rng)
        metrics["sources"][name] = {"cases": len(cases), "hidden_title_eligible_share": eligible_share, "scorers": agg}
        base = per.get("collection_profile_v1") or []
        for sname, a in agg.items():
            rows.append({"source": name, "scorer": sname, **{k: a[k]["mean"] for k in a}, "cases": a["hits@8"]["n"]})
            if sname != "collection_profile_v1" and base:
                metrics["sources"][name].setdefault("paired_vs_profile_v1", {})[sname] = paired_bootstrap_diff(
                    [m["hits@8"] for m in per[sname]], [m["hits@8"] for m in base])
        if name.endswith(":val") and "collection_profile_v1" in agg:
            b = agg["collection_profile_v1"]
            for sname, a in agg.items():
                if sname in ("collection_profile_v1", "manual_slides_v0"):
                    continue
                ok = a["auc_sampled50"]["mean"] >= 0.65 and a["hits@8"]["mean"] - b["hits@8"]["mean"] >= 0.02
                # a proxy benchmark (liked history) does not measure collection-taste matching: no decision
                decision = "n/a (proxy benchmark)" if name.startswith("proxy") else bool(ok)
                decisions.append({"source": name, "candidate": sname, "replace_default": decision,
                                  "val_auc_sampled50": round(a["auc_sampled50"]["mean"], 4),
                                  "delta_hits@8": round(a["hits@8"]["mean"] - b["hits@8"]["mean"], 4)})
    metrics["replacement_decisions_on_validation"] = decisions
    run.comparison(rows)
    lines = [f"# Collection recommendations (algorithm 2) - {settings.mode} data", "",
             f"Dataset `{ctx.version}`. Leave-one-out cases; K={K}; the rule layer (served, recommendable, not already "
             "in the visible collection) is identical for every scorer, positives and negatives.", ""]
    if settings.mode == "synthetic":
        lines += ["> Synthetic collections come from a generator process; results validate the implementation only.", ""]
    lines += [md_table(rows, ["source", "scorer", "hits@8", "ndcg@8", "auc_full_pool", "auc_sampled50", "cases"]), ""]
    if decisions:
        lines += ["## Replacement rule on validation (AUC >= 0.65 and Hits@8 +2pp vs collection_profile_v1)", "",
                  md_table(decisions, ["source", "candidate", "val_auc_sampled50", "delta_hits@8", "replace_default"]), ""]
    lines += ["AUC against sampled unobserved titles treats them as negatives only for measurement; they are not "
              "confirmed dislikes. `manual_slides_v0` is the slides' formula kept verbatim as the baseline.", ""]
    for key in ("gbdt_proxy_training", "gbdt_training"):
        imp = (metrics.get(key) or {}).get("permutation_importance_val")
        if imp:
            top = ", ".join(f"{d['feature']} {d['importance']:.3f}" for d in imp[:4])
            lines += [f"GBDT permutation importance on validation ({key}): {top}.", ""]
            if key == "gbdt_proxy_training" and imp[0]["feature"] == "train_popularity":
                lines += ["On the proxy benchmark the GBDT's advantage comes almost entirely from train popularity: the "
                          "hidden titles are titles the user rated >= 8, which are mostly popular. This says nothing about "
                          "matching a collection's taste, so the proxy result does not change the collection default.", ""]
    lines += [f"- note: {n}" for n in run.notes]
    return run.finish(metrics, "\n".join(lines) + "\n", {"sources": [s for s, _, _ in sources]})
