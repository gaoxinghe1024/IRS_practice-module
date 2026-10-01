"""Homepage experiment (algorithm 3): recall engine x list optimiser on the ranking protocol.

Protocol
  * users: seeded sample of ranking-protocol users with a held-out test positive (rating >= 8)
  * history: that user's train ratings only; candidates: served, recommendable titles not in the
    history and not the user's validation positive (the other held-out item) -> full-catalogue ranking
  * engines: ICF (exact adjusted cosine, pure: unsupported titles are not ranked), MF, NCF, popularity
  * optimisers over the engine's top-`candidate_k` pool (same pool and scores for every optimiser):
    no-rerank (recall order), relevance-greedy, GA (several seeds)
  * tuning: the ICF prediction damping is chosen on the VALIDATION positive of a disjoint user sample;
    test positives are only used for the final report
  * MAE/RMSE are properties of the recall predictor and are reported by the `explicit` experiment
"""

from __future__ import annotations

import time

import numpy as np

from app.recommendation.rerank.ga import GeneticListOptimizer, relevance_greedy
from app.recommendation.rerank.objective import ListObjective, fresh_flags, novelty_scores
from evaluation.common import Run, ci_text, md_table
from evaluation.metrics import (
    bootstrap_ci,
    hits_at_k,
    intra_list_diversity,
    mrr_at_k,
    ndcg_binary,
    paired_bootstrap_diff,
)
from evaluation.protocols import engine_scores, load_context, sample_users
from pipelines.split import load_split

K = 10


def _objective(ctx, pool_ids, pool_scores, hist_tags, weights):
    rel = pool_scores - pool_scores.min()
    rel = rel / rel.max() if rel.max() > 0 else np.ones_like(rel)
    sim, valid = ctx.weighted.pairwise(ctx.snap, pool_ids)
    return ListObjective(rel, sim, valid, novelty_scores(ctx.snap.train_interactions, pool_ids),
                         fresh_flags(ctx.snap, pool_ids, hist_tags), weights)


def _list_metrics(ctx, sel_ids, target, hist_tags):
    sim, valid = ctx.weighted.pairwise(ctx.snap, sel_ids)
    nov = novelty_scores(ctx.snap.train_interactions, sel_ids)
    fr = fresh_flags(ctx.snap, sel_ids, hist_tags)
    ranked = sel_ids.tolist()
    return {"hits@5": hits_at_k(ranked, {target}, 5), "hits@10": hits_at_k(ranked, {target}, 10),
            "mrr@10": mrr_at_k(ranked, {target}, 10), "ndcg@10": ndcg_binary(ranked, {target}, 10),
            "ild": intra_list_diversity(sim, valid), "novelty": float(np.mean(nov)) if nov is not None else float("nan"),
            "fresh": float(np.mean(fr)) if fr is not None else float("nan")}


SHORT_TUNE = (5, 8, 15)                  # history lengths (plus the full history) used to choose k
SHORT_REPORT = (1, 3, 5, 8, 15, 30)      # history lengths in the short-history (warm-gate) analysis


def _truncate(items, r, n, user):
    """A seeded random subset of n train ratings: what a new user who rated n titles would look like."""
    if n is None or n >= len(items):
        return items, r
    pick = np.sort(np.random.default_rng(1000003 * int(user) + n).choice(len(items), size=n, replace=False))
    return items[pick], r[pick]


def _val_hits(ctx, users, hists, val_pos, n=None) -> list[float]:
    hits = []
    for u in users:
        if u not in hists or u not in val_pos:
            continue
        items, r = hists[u]
        if n is not None and len(items) < n:
            continue
        mask = ctx.pool_mask.copy()
        mask[items] = False                      # the full history stays excluded from candidates
        ti, tr = _truncate(items, r, n, u)
        ids, _ = engine_scores(ctx, "icf", u, ti, tr, mask)
        hits.append(hits_at_k(ids[:K].tolist(), {val_pos[u]}, K))
    return hits


def tune_damping(ctx, users, hists, val_pos, grid=(0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0)) -> dict:
    out = {}
    base = ctx.icf.damping
    for d in grid:
        ctx.icf.damping = d
        hits = _val_hits(ctx, users, hists, val_pos)
        out[d] = float(np.mean(hits)) if hits else float("nan")
    ctx.icf.damping = base
    return out


def tune_prior(ctx, users, hists, val_pos, grid=(0.0, 2.0, 5.0, 10.0, 20.0)) -> dict:
    """Validation hits@10 per user-mean prior weight k, for short histories and the full history."""
    out = {}
    base = ctx.icf.prior_weight
    for k in grid:
        ctx.icf.prior_weight = k
        per_n = {str(n): float(np.mean(h)) if (h := _val_hits(ctx, users, hists, val_pos, n)) else float("nan")
                 for n in SHORT_TUNE}
        full = _val_hits(ctx, users, hists, val_pos)
        per_n["full"] = float(np.mean(full)) if full else float("nan")
        per_n["mean"] = float(np.nanmean(list(per_n.values())))
        out[k] = per_n
    ctx.icf.prior_weight = base
    return out


CONTENT_TUNE = (1, 3, 5)                 # cold / partial regime where the home feed relies on content scores


def _content_list(ctx, ti, tr, mask, weight):
    """Home-feed content ranking (services/recommend.py:home_content_scores) for a truncated history."""
    from app.recommendation.profiles import UserState
    from app.services.recommend import content_profile, home_content_scores

    us = UserState(user_id=None, profile_version=0, ratings={int(i): float(x) for i, x in zip(ti, tr, strict=True)})
    prof = content_profile(ctx.snap, us)
    pool = np.flatnonzero(mask)
    if prof is None or not len(pool):
        return None, {}
    cs = home_content_scores(ctx.snap, prof, pool, weight)
    order = np.lexsort((ctx.snap.keys[pool], -cs))
    return pool[order], dict(zip(pool.tolist(), cs.tolist(), strict=True))


def tune_content_weight(ctx, users, hists, val_pos, grid=(0.0, 0.25, 0.5, 0.75, 0.9)) -> dict:
    """Validation hits@10 of the home content ranking per popularity weight, for histories of 1 / 3 / 5 ratings."""
    out = {}
    for w in grid:
        per_n = {}
        for n in CONTENT_TUNE:
            hits = []
            for u in users:
                if u not in hists or u not in val_pos:
                    continue
                items, r = hists[u]
                if len(items) < n:
                    continue
                mask = ctx.pool_mask.copy()
                mask[items] = False
                ti, tr = _truncate(items, r, n, u)
                ids, _ = _content_list(ctx, ti, tr, mask, w)
                if ids is not None:
                    hits.append(hits_at_k(ids[:K].tolist(), {val_pos[u]}, K))
            per_n[str(n)] = float(np.mean(hits)) if hits else float("nan")
        per_n["mean"] = float(np.nanmean(list(per_n.values())))
        out[w] = per_n
    return out


def short_history_analysis(ctx, users, hists, test_pos, val_pos, gate: int, exact_icf, run,
                           content_weight: float = 0.0) -> dict:
    """Test hits@10 by history length: ICF (served / exact), content profile, popularity, served blend."""
    served_icf = ctx.icf
    res: dict = {}
    for n in SHORT_REPORT:
        rows = {m: [] for m in ("icf_served", "icf_exact", "content", "content_tags_only", "popularity",
                                "served_blend")}
        for u in users:
            if u not in hists or u not in test_pos:
                continue
            items, r = hists[u]
            if len(items) < n:
                continue
            mask = ctx.pool_mask.copy()
            mask[items] = False
            if u in val_pos:
                mask[val_pos[u]] = False
            target = test_pos[u]
            if not mask[target]:
                continue
            ti, tr = _truncate(items, r, n, u)
            lists = {}
            ctx.icf = served_icf
            ids_s, sc_s = engine_scores(ctx, "icf", u, ti, tr, mask)
            lists["icf_served"] = ids_s
            ctx.icf = exact_icf
            lists["icf_exact"], _ = engine_scores(ctx, "icf", u, ti, tr, mask)
            ctx.icf = served_icf
            lists["popularity"], _ = engine_scores(ctx, "popularity", u, ti, tr, mask)
            cids, cont = _content_list(ctx, ti, tr, mask, content_weight)
            lists["content"] = cids if cids is not None else lists["popularity"]
            if content_weight > 0:
                pids, _ = _content_list(ctx, ti, tr, mask, 0.0)
                lists["content_tags_only"] = pids if pids is not None else lists["popularity"]
            # the served blend (services/recommend.py:home): warm >= gate -> ICF; partial -> lambda mix
            if n >= gate and len(ids_s):
                lists["served_blend"] = ids_s
            else:
                lam = n / gate
                cf_top = ids_s[:20]
                cf_norm = {int(i): (float(v) - 1.0) / 9.0 for i, v in zip(cf_top, sc_s[:20], strict=True)}
                c_top = lists["content"][:20].tolist()
                union = set(cf_norm) | set(int(i) for i in c_top)
                blend = {i: lam * cf_norm.get(i, 0.0) + (1 - lam) * cont.get(i, 0.0) for i in union}
                lists["served_blend"] = np.array(sorted(blend, key=lambda i: (-blend[i], ctx.snap.keys[i])))
            for m, ids in lists.items():
                rows[m].append(hits_at_k(np.asarray(ids)[:K].tolist(), {target}, K))
        res[str(n)] = {m: bootstrap_ci(v) for m, v in rows.items() if v}
    ctx.icf = served_icf
    return res


def run(settings, sample_users_n: int | None = None, seeds: int = 3, candidate_k: int = 20, tag=None) -> dict:
    ctx = load_context(settings)
    sample_users_n = sample_users_n or (1500 if settings.mode == "real" else None)
    run = Run(settings, "homepage", {"sample_users": sample_users_n, "ga_seeds": seeds, "candidate_k": candidate_k,
                                     "k": K, "tag": tag, "tuning": "20% of eligible users (max 600), disjoint"})
    test = load_split(settings, "ranking", "test", ctx.version)
    val = load_split(settings, "ranking", "val", ctx.version)
    test_pos = dict(zip(test.offline_user_id.astype(int), test.item_idx.astype(int), strict=True))
    val_pos = dict(zip(val.offline_user_id.astype(int), val.item_idx.astype(int), strict=True))
    # disjoint, deterministic partition: 20% of eligible users (max 600) tune on validation, the rest are evaluated
    everyone = sample_users(test, None, seed=11)
    order = np.random.default_rng(11).permutation(len(everyone))
    n_tune = min(600, max(1, len(everyone) // 5))
    tune_users = sorted(everyone[k] for k in order[:n_tune])
    rest = [everyone[k] for k in order[n_tune:]]
    eval_users = sorted(rest if sample_users_n is None else rest[:sample_users_n])
    ga_cfg = dict(settings.recommendation.get("ga", {}))
    weights = ga_cfg.pop("weights", {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2, "fresh": 0.1})
    ga_cfg["time_budget_ms"] = None          # offline: fixed generation budget -> reproducible
    engines = [e for e, m in (("icf", ctx.icf), ("mf", ctx.mf), ("ncf", ctx.ncf)) if m is not None] + ["popularity"]

    # ---- validation-only tuning on a disjoint user sample: damping (full histories), then the user-mean
    # prior weight k (mean over history lengths 5 / 8 / 15 / full, so the new-user regime counts)
    tuning, prior_tuning, exact_icf = None, None, None
    if ctx.icf is not None:
        from app.recommendation.recall.icf import ICFScorer

        exact_icf = ICFScorer(ctx.icf.model)                  # the brief's exact formula, for comparison
        tune_pool = tune_users
        th = ctx.histories(tune_pool)
        ctx.icf.prior_weight = 0.0
        tuning = tune_damping(ctx, tune_pool, th, val_pos)
        best = max(tuning, key=lambda d: (tuning[d], -d))
        ctx.icf.damping = best
        run.notes.append(f"ICF damping tuned on validation positives of {len(tune_pool)} other users: {tuning} -> {best}")
        if best == max(tuning):
            run.notes.append("the selected damping is the largest value in the grid; a larger value might score higher")
        prior_tuning = tune_prior(ctx, tune_pool, th, val_pos)
        best_k = max(prior_tuning, key=lambda k: (prior_tuning[k]["mean"], -k))
        ctx.icf.prior_weight = best_k
        run.notes.append(f"ICF user-mean prior weight tuned on the same validation users (mean hits@10 over history "
                         f"lengths {SHORT_TUNE} and full): {prior_tuning} -> {best_k}")
    content_tuning, content_w = None, float(settings.recommendation.get("home", {}).get("content_popularity_weight", 0.0))
    if tune_users:
        th = ctx.histories(tune_users)
        content_tuning = tune_content_weight(ctx, tune_users, th, val_pos)
        content_w = max(content_tuning, key=lambda w: (content_tuning[w]["mean"], -w))
        run.notes.append(f"Home content popularity weight tuned on validation (histories of {CONTENT_TUNE} ratings): "
                         f"{content_tuning} -> {content_w}")

    hists = ctx.histories(eval_users)
    per = {}          # (engine, optimiser) -> list of metric dicts (aligned by user order)
    recall_at = {e: {20: [], 50: [], 100: []} for e in engines}
    served_lists = {}
    timings = {}
    aliases = {}
    preds = []
    for u in eval_users:
        if u not in hists:
            run.failure({"user": u, "reason": "no train history"})
            continue
        items, r = hists[u]
        target = test_pos[u]
        mask = ctx.pool_mask.copy()
        mask[items] = False
        if u in val_pos:
            mask[val_pos[u]] = False
        if not mask[target]:
            run.failure({"user": u, "reason": "test positive not in candidate universe (not recommendable)"})
            continue
        hist_tags = ctx.history_tags(items)
        for e in engines:
            t0 = time.perf_counter()
            ids, scores = engine_scores(ctx, e, u, items, r, mask)
            timings.setdefault(e, []).append((time.perf_counter() - t0) * 1000)
            for n in (20, 50, 100):
                recall_at[e][n].append(float(target in set(ids[:n].tolist())))
            pool_ids, pool_s = ids[:candidate_k], scores[:candidate_k]
            if len(pool_ids) == 0:
                for opt in ("no_rerank", "greedy") + tuple(f"ga_s{s}" for s in range(seeds)):
                    per.setdefault((e, opt), []).append(None)
                continue
            obj = _objective(ctx, pool_ids, pool_s, hist_tags, weights)
            no_rr = np.arange(min(K, len(pool_ids)))
            greedy = relevance_greedy(obj, K).selected
            results = {"no_rerank": no_rr, "greedy": greedy}
            aliases.setdefault(e, []).append(bool(np.array_equal(no_rr, greedy)))
            for s in range(seeds):
                t1 = time.perf_counter()
                res = GeneticListOptimizer(**ga_cfg).optimize(obj, K, np.random.default_rng(1000 + s))
                timings.setdefault(f"ga:{e}", []).append((time.perf_counter() - t1) * 1000)
                results[f"ga_s{s}"] = res.selected
            for opt, sel in results.items():
                m = _list_metrics(ctx, pool_ids[sel], target, hist_tags)
                m["fitness"] = float(obj.fitness(np.asarray(sel)[None, :])[0])
                per.setdefault((e, opt), []).append(m)
                served_lists.setdefault((e, opt), set()).update(pool_ids[sel].tolist())
                if opt in ("greedy", "ga_s0"):
                    preds.append({"user": u, "engine": e, "optimizer": opt, "target": str(ctx.snap.keys[target]),
                                  "list": "|".join(str(ctx.snap.keys[i]) for i in pool_ids[sel])})
        run.case({"user": u, "target": str(ctx.snap.keys[target]), "history_size": len(items),
                  **{f"{e}:{o}:hits@10": (per[(e, o)][-1] or {}).get("hits@10") for (e, o) in per if o in ("greedy", "ga_s0")}})
    run.predictions(preds)
    short = None
    if ctx.icf is not None:
        gate = int(settings.recommendation.get("home", {}).get("warm_gate", 8))
        short = short_history_analysis(ctx, eval_users[:500], hists, test_pos, val_pos, gate, exact_icf, run,
                                       content_weight=content_w)

    # ---- aggregate
    n_pool = int(ctx.pool_mask.sum())
    rows, metrics = [], {"engines": {}, "tuning_icf_damping_val_hits@10": tuning,
                         "tuning_icf_prior_weight_val_hits@10": prior_tuning, "short_history_test_hits@10": short,
                         "icf_served_params": ({"damping": ctx.icf.damping, "user_mean_prior_weight": ctx.icf.prior_weight}
                                               if ctx.icf is not None else None),
                         "tuning_home_content_popularity_weight_val_hits@10": content_tuning,
                         "home_served_params": {"content_popularity_weight": content_w}, "users_evaluated": 0}
    for e in engines:
        metrics["engines"][e] = {"candidate_recall": {str(n): bootstrap_ci(v) for n, v in recall_at[e].items()},
                                 "no_rerank_equals_greedy_share": float(np.mean(aliases.get(e, [np.nan]))),
                                 "recall_latency_ms_p50": float(np.percentile(timings.get(e, [np.nan]), 50)),
                                 "ga_latency_ms_p50": float(np.percentile(timings.get(f"ga:{e}", [np.nan]), 50))
                                 if timings.get(f"ga:{e}") else None}
        for opt in ["no_rerank", "greedy"] + [f"ga_s{s}" for s in range(seeds)]:
            vals = [m for m in per.get((e, opt), []) if m is not None]
            if not vals:
                continue
            agg = {k: bootstrap_ci([m[k] for m in vals]) for k in vals[0]}
            agg["catalog_coverage"] = round(len(served_lists.get((e, opt), set())) / max(1, n_pool), 5)
            metrics["engines"][e][opt] = agg
            if opt in ("no_rerank", "greedy", "ga_s0"):
                rows.append({"engine": e, "optimizer": opt if opt != "ga_s0" else "ga",
                             **{k: agg[k]["mean"] for k in ("hits@5", "hits@10", "mrr@10", "ndcg@10", "ild", "novelty",
                                                            "fresh", "fitness")},
                             "coverage": agg["catalog_coverage"], "n": agg["hits@10"]["n"]})
        g = [m for m in per.get((e, "greedy"), [])]
        a = [m for m in per.get((e, "ga_s0"), [])]
        pair = [(x, y) for x, y in zip(g, a, strict=True) if x is not None and y is not None]
        if pair:
            metrics["engines"][e]["ga_minus_greedy"] = {
                k: paired_bootstrap_diff([y[k] for _, y in pair], [x[k] for x, _ in pair])
                for k in ("hits@10", "mrr@10", "ild", "novelty", "fresh", "fitness")}
            seeds_h = [np.mean([m["hits@10"] for m in per[(e, f"ga_s{s}")] if m]) for s in range(seeds)]
            metrics["engines"][e]["ga_seed_hits@10"] = [round(float(x), 5) for x in seeds_h]
    metrics["users_evaluated"] = len(per.get((engines[0], "greedy"), []))
    run.comparison(rows)
    report = _report(settings, ctx, metrics, rows, engines, seeds, candidate_k, run)
    return run.finish(metrics, report, {"eval_users": len(eval_users), "candidate_universe": n_pool})


def _report(settings, ctx, metrics, rows, engines, seeds, candidate_k, run) -> str:
    lines = [f"# Homepage experiment (algorithm 3) - {settings.mode} data", "",
             f"Dataset version `{ctx.version}`; ranking protocol (one held-out high-rated test title per user). "
             f"Users evaluated: {metrics['users_evaluated']}. Candidate universe: served, recommendable titles not "
             f"in the user's train history (validation positive removed). Candidate pool for optimisers: top {candidate_k} "
             f"of each engine; list length K={K}; GA seeds: {seeds}.", ""]
    if settings.mode == "synthetic":
        lines += ["> Synthetic data: these numbers validate that the pipeline and algorithms run correctly on a "
                  "generated world; they are not evidence of real recommendation quality.", ""]
    lines += ["## Main table (means; 95% bootstrap CIs in metrics.json)", "",
              md_table(rows, ["engine", "optimizer", "hits@5", "hits@10", "mrr@10", "ndcg@10", "ild", "novelty", "fresh",
                              "fitness", "coverage", "n"]), ""]
    lines += ["`no_rerank` is the recall order truncated to K. When relevance is a monotone transform of the recall "
              "score, relevance-greedy selects the same list; the share of users where they coincide is reported "
              "per engine below (an alias, not an independent result).", ""]
    for e in engines:
        m = metrics["engines"][e]
        lines.append(f"### {e}")
        lines.append("- candidate recall@20/50/100: " + ", ".join(
            ci_text(m['candidate_recall'][n]) for n in ("20", "50", "100")))
        lines.append(f"- no-rerank == greedy for {m['no_rerank_equals_greedy_share']:.1%} of users")
        if "ga_minus_greedy" in m:
            d = m["ga_minus_greedy"]
            lines.append("- GA - greedy (paired bootstrap): " + "; ".join(
                f"{k} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in d.items()))
            lines.append(f"- GA hits@10 by seed: {m.get('ga_seed_hits@10')}")
        lines.append(f"- latency p50: recall {m['recall_latency_ms_p50']:.1f} ms"
                     + (f", GA {m['ga_latency_ms_p50']:.1f} ms" if m.get("ga_latency_ms_p50") else ""))
        lines.append("")
    if metrics.get("tuning_icf_damping_val_hits@10"):
        lines += ["## Validation-only tuning", "", f"ICF prediction damping (validation hits@10, full histories): "
                  f"{metrics['tuning_icf_damping_val_hits@10']}", ""]
    if metrics.get("tuning_icf_prior_weight_val_hits@10"):
        pt = metrics["tuning_icf_prior_weight_val_hits@10"]
        cols = [str(n) for n in SHORT_TUNE] + ["full", "mean"]
        lines += ["ICF user-mean prior weight k (validation hits@10 by history length; chosen by the mean):", "",
                  md_table([{"k": k, **{c: round(v[c], 4) for c in cols}} for k, v in pt.items()], ["k"] + cols), "",
                  f"Served ICF parameters used for the test results above: {metrics['icf_served_params']}.", ""]
    if metrics.get("tuning_home_content_popularity_weight_val_hits@10"):
        ct = metrics["tuning_home_content_popularity_weight_val_hits@10"]
        cols = [str(n) for n in CONTENT_TUNE] + ["mean"]
        lines += ["Home content ranking: popularity weight g in (1 - g) * profile match + g * popular_quality "
                  "(validation hits@10 by history length; chosen by the mean):", "",
                  md_table([{"g": w, **{c: round(v[c], 4) for c in cols}} for w, v in ct.items()], ["g"] + cols), "",
                  f"Served home parameters used below: {metrics['home_served_params']}.", ""]
    if metrics.get("short_history_test_hits@10"):
        sh = metrics["short_history_test_hits@10"]
        methods = ["icf_served", "icf_exact", "content", "content_tags_only", "popularity", "served_blend"]
        trows = []
        for n, m in sh.items():
            trows.append({"ratings": n, **{k: (round(m[k]["mean"], 4) if k in m else None) for k in methods},
                          "users": max((m[k]["n"] for k in m), default=0)})
        lines += ["## Short histories (warm-gate analysis, test positives)", "",
                  "Each user's train history is cut to a seeded random subset of n ratings (a new user who rated n "
                  "titles); the rest of the history stays excluded from the candidates. `icf_exact` is the brief's "
                  "formula (no damping, no prior); `content` is the home content ranking with the tuned popularity "
                  "weight and `content_tags_only` the pure tag profile; `served_blend` mirrors the product: below the warm gate "
                  f"({int(settings.recommendation.get('home', {}).get('warm_gate', 8))} ratings) a lambda = n / gate "
                  "mix of ICF and the content profile, above it ICF. Hits@10, first 500 evaluation users.", "",
                  md_table(trows, ["ratings"] + methods + ["users"]), ""]
    lines += ["## Reading the results", "",
              "- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside "
              "the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and "
              "diversity trade-offs are shown side by side.",
              "- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).",
              "- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.",
              ""] + [f"- note: {n}" for n in run.notes]
    return "\n".join(lines) + "\n"
