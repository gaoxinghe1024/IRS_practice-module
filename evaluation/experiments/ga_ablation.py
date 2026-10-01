"""GA ablations (validation users only; test positives are not touched).

  * objective ablations: full weights vs dropping diversity / novelty / fresh
  * candidate pool size 20 / 50 / 100 (same engine scores)
  * baselines on the same pool: relevance-greedy, MMR (lambda 0.7, a separately named method)
  * exhaustive oracle on small pools (12 candidates, K=5): optimality gap of the GA
  * GA seed variance
"""

from __future__ import annotations

import numpy as np

from app.recommendation.rerank.ga import GeneticListOptimizer, exhaustive, mmr, relevance_greedy
from app.recommendation.rerank.objective import ListObjective, fresh_flags, novelty_scores
from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci, hits_at_k, intra_list_diversity
from evaluation.protocols import engine_scores, load_context, sample_users
from pipelines.split import load_split

K = 10
FULL = {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2, "fresh": 0.1}
VARIANTS = {"full": FULL, "no_diversity": {"relevance": 0.4, "novelty": 0.2, "fresh": 0.1},
            "no_novelty": {"relevance": 0.4, "diversity": 0.3, "fresh": 0.1},
            "no_fresh": {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2}}


def run(settings, sample_users_n: int | None = None, seeds: int = 3, tag=None, **_) -> dict:
    ctx = load_context(settings)
    engine = "icf" if ctx.icf is not None else ("mf" if ctx.mf is not None else "popularity")
    n = sample_users_n or (600 if settings.mode == "real" else None)
    run = Run(settings, "ga_ablation", {"engine": engine, "sample_users": n, "seeds": seeds, "tag": tag})
    val = load_split(settings, "ranking", "val", ctx.version)
    test = load_split(settings, "ranking", "test", ctx.version)
    vpos = dict(zip(val.offline_user_id.astype(int), val.item_idx.astype(int), strict=True))
    tpos = dict(zip(test.offline_user_id.astype(int), test.item_idx.astype(int), strict=True))
    users = sample_users(val, n, seed=41)
    hists = ctx.histories(users)
    ga_cfg = {k: v for k, v in settings.recommendation.get("ga", {}).items() if k != "weights"}
    ga_cfg["time_budget_ms"] = None
    res: dict[str, list[dict]] = {}
    gaps, oracle_n = [], 0
    for u in users:
        if u not in hists:
            continue
        items, r = hists[u]
        target = vpos[u]
        mask = ctx.pool_mask.copy()
        mask[items] = False
        if u in tpos:
            mask[tpos[u]] = False            # the test positive is not a candidate during validation
        if not mask[target]:
            continue
        ids, scores = engine_scores(ctx, engine, u, items, r, mask)
        tags = ctx.history_tags(items)
        for ck in (20, 50, 100):
            pool, s = ids[:ck], scores[:ck]
            if len(pool) < K + 1:
                continue
            rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones(len(s))
            sim, valid = ctx.weighted.pairwise(ctx.snap, pool)
            nov = novelty_scores(ctx.snap.train_interactions, pool)
            fr = fresh_flags(ctx.snap, pool, tags)
            for vname, w in (VARIANTS.items() if ck == 20 else [("full", FULL)]):
                obj = ListObjective(rel, sim, valid, nov, fr, w)
                methods = {"greedy": relevance_greedy(obj, K).selected, "mmr_0.7": mmr(obj, K, 0.7).selected}
                for sd in range(seeds if (ck == 20 and vname == "full") else 1):
                    methods[f"ga_s{sd}"] = GeneticListOptimizer(**ga_cfg).optimize(obj, K, np.random.default_rng(sd)).selected
                full_obj = ListObjective(rel, sim, valid, nov, fr, FULL)
                for mname, sel in methods.items():
                    sel_ids = pool[sel]
                    ss, vv = ctx.weighted.pairwise(ctx.snap, sel_ids)
                    m = {"hits@10": hits_at_k(sel_ids.tolist(), {target}, K), "ild": intra_list_diversity(ss, vv),
                         "novelty": float(np.mean(nov[sel])) if nov is not None else float("nan"),
                         "fresh": float(np.mean(fr[sel])) if fr is not None else float("nan"),
                         "fitness_full_objective": float(full_obj.fitness(np.asarray(sel)[None, :])[0])}
                    res.setdefault(f"pool{ck}:{vname}:{mname}", []).append(m)
        # optimality gap on a small pool
        pool, s = ids[:12], scores[:12]
        if len(pool) == 12:
            rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones(12)
            sim, valid = ctx.weighted.pairwise(ctx.snap, pool)
            obj = ListObjective(rel, sim, valid, novelty_scores(ctx.snap.train_interactions, pool),
                                fresh_flags(ctx.snap, pool, tags), FULL)
            best = exhaustive(obj, 5).fitness
            got = GeneticListOptimizer(**ga_cfg).optimize(obj, 5, np.random.default_rng(0)).fitness
            gaps.append(best - got)
            oracle_n += 1
    rows = []
    metrics = {"engine": engine, "users": len(hists), "results": {}}
    for key, vals in sorted(res.items()):
        agg = {k: bootstrap_ci([v[k] for v in vals]) for k in vals[0]}
        metrics["results"][key] = agg
        pool_s, variant, method = key.split(":")
        rows.append({"pool": pool_s, "objective": variant, "method": method,
                     **{k: agg[k]["mean"] for k in agg}, "n": agg["hits@10"]["n"]})
    metrics["oracle"] = {"pools": oracle_n, "mean_gap": float(np.mean(gaps)) if gaps else None,
                         "max_gap": float(np.max(gaps)) if gaps else None,
                         "share_optimal": float(np.mean(np.array(gaps) < 1e-9)) if gaps else None}
    run.comparison(rows)
    lines = [f"# GA ablation (validation users) - {settings.mode} data", "",
             f"Engine `{engine}`, {metrics['users']} validation users (their test positive excluded from candidates). "
             "All methods in a row group use the same candidate pool and scores.", ""]
    if settings.mode == "synthetic":
        lines += ["> Synthetic data: validates the optimiser implementation and trade-offs, not real quality.", ""]
    lines += [md_table(rows, ["pool", "objective", "method", "hits@10", "ild", "novelty", "fresh",
                              "fitness_full_objective", "n"]), "",
              f"Exhaustive oracle on 12-candidate pools (K=5): {oracle_n} pools, mean fitness gap "
              f"{metrics['oracle']['mean_gap']}, max gap {metrics['oracle']['max_gap']}, optimal share "
              f"{metrics['oracle']['share_optimal']}.", "",
              "MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool."]
    return run.finish(metrics, "\n".join(lines) + "\n", {"engine": engine})
