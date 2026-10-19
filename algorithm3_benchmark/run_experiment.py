# -*- coding: utf-8 -*-
"""
run_experiment.py — main experiment: produces all comparison data for the report.

Two-phase evaluation over a 10-tier candidate pool x 2 sampling seeds x 20,000 users
on 3 recall engines:

  phase A (cheap):  no-rerank ranks of the hidden item within each candidate pool
                    -> HR@5 / HR@10 (all data for report part 1)
  phase B (costly): four rerank arms on the per-tier winner engine
                    -> HR@5 / HR@10 + list-level metrics
                       (diversity / novelty / freshness / long-tail share)
                    -> all data for report part 2 ("what GA actually did")

The four rerank arms share the same candidate pool and recall result; only the
selection rule differs: greedy (= no rerank), randw (random within top-20 by
relevance), randp (random from the whole pool), GA (weighted optimization).

Outputs:
    results/ranks.npz        per-user raw data (sole data source for the report)
    results/experiment.json  per-tier summary + no-rerank HR per engine + protocol

Reproducibility: ICF is deterministic and must reproduce stored ranks bit-for-bit;
GA uses default_rng([seed, pool_size, engine, user]); the random control arms use
an independent arm_rng so adding arms never perturbs published GA numbers.
"""
import argparse
import json
import os
import platform
import time

import numpy as np

import config as C
import data_loader as D
import engine_registry as REG
import candidate_pool as CP
import list_metrics as LM
from rerank_ga import rerank_ga
from rerank_greedy import rerank_greedy
from rerank_random import rerank_random

ENGINES = REG.ENGINES
SEEDS = [CP.SAMPLING_SEED_BASE, CP.SAMPLING_SEED_BASE + 1]   # [7042, 7043]

GA_METRICS = ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")
GREEDY_METRICS = ("nov5", "nov10", "div10", "fresh10", "lt10")
RANDOM_METRICS = ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")



def log(msg):
    print(msg, flush=True)


def _arm_metrics(items, tag_sets, novelty, fresh, nov_thr, hidden):
    """Compute the 7 metrics for one Top-10 list; shared verbatim by all four arms."""
    met = LM.list_metrics(items, tag_sets, novelty, fresh)
    K = C.HR_K
    return dict(
        hr5=1.0 if hidden in items[:K] else 0.0,
        hr10=1.0 if hidden in items else 0.0,
        nov5=float(np.mean([novelty[i] for i in items[:K]])),
        nov10=met["novelty"],
        div10=met["diversity"],
        fresh10=met["fresh"],
        lt10=float(np.mean([novelty[i] >= nov_thr for i in items])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=C.COHORT_SIZE)
    ap.add_argument("--chunk", type=int, default=CP.SCORE_CHUNK)
    ap.add_argument("--prefix", type=int, default=0,
                    help="if >0, evaluate only the first N users of the real cohort "
                         "(smoke test / bit-level verification)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files (e.g. --tag smoke writes ranks_smoke.npz), "
                         "so smoke runs do not overwrite official results")
    args = ap.parse_args()

    t_all = time.time()
    grid, grid_label = CP.GRID, CP.GRID_LABEL
    n_grid = len(grid)

    log("=" * 78)
    log("ICF + GA main experiment: 10-tier candidate pool x 3 recall engines x 4 rerank arms")
    log("=" * 78)

    # ------------------------------------------------------------ 1. data
    log("\n[1/5] Loading data ...")
    R = D.load_ratings_train()
    a2i, u2i = D.load_mappings()
    n_users, n_items = R.shape
    val = D.load_validation()
    v = D.build_validation_indexed(val, a2i, u2i)
    hidden_all = D.build_eligible_cohort(R, v)
    cohort = D.sample_cohort(hidden_all, size=args.users)
    if args.prefix:
        # --prefix takes the FIRST N of the real 20,000-user cohort (not a fresh sample),
        # so smoke results can be compared user-by-user against the full run.
        cohort = cohort.iloc[:args.prefix].reset_index(drop=True)
        log(f"  --prefix {args.prefix}: evaluating only the first {args.prefix} users of the real cohort")
    cohort_u = cohort["u"].to_numpy().astype(np.int64)
    hidden_i = cohort["i"].to_numpy().astype(np.int64)
    n = len(cohort_u)
    log(f"  cohort={n:,}  items={n_items:,}  train ratings={R.nnz:,}")

    feats = D.load_item_features(a2i, n_items)
    years = D.load_release_years(a2i, n_items)
    tag_sets = LM.build_tag_sets(feats)
    novelty = feats["novelty"]
    fresh, fresh_thr = LM.fresh_flag(years, span=5)
    seen = feats["seen"]
    nov_thr = float(np.median(novelty[seen]))
    log(f"  long-tail threshold = median novelty of items with metadata = {nov_thr:.4f}")

    perms = {s: CP.pool_permutation(s, n_items) for s in SEEDS}

    # ------------------------------------------------------------ 2. engines
    log("\n[2/5] Training engines (ICF deterministic; SVD/NCF retrained with fixed seed) ...")
    engines = REG.load_engines(ENGINES, R, n_items)

    ns = len(SEEDS)
    rank_nr = {e: np.zeros((ns, n_grid, n), np.int32) for e in ENGINES}
    cand_count = np.zeros((ns, n_grid, n), np.int32)
    ga_m = {e: {m: np.zeros((ns, n_grid, n), np.float32) for m in GA_METRICS}
            for e in ENGINES}
    gr_m = {e: {m: np.zeros((ns, n_grid, n), np.float32) for m in GREEDY_METRICS}
            for e in ENGINES}
    randw_m = {m: np.zeros((ns, n_grid, n), np.float32) for m in RANDOM_METRICS}
    randp_m = {m: np.zeros((ns, n_grid, n), np.float32) for m in RANDOM_METRICS}

    # ------------------------------------------------------ phase A: no-rerank ranks
    log("\n[3/5] phase A: no-rerank ranks (source of HR@5 / HR@10) ...")
    t0 = time.time()
    for s0 in range(0, n, args.chunk):
        e0 = min(s0 + args.chunk, n)
        uu = cohort_u[s0:e0]
        hi = hidden_i[s0:e0]
        m = e0 - s0
        rows = []
        for r in range(m):
            a, b = R.indptr[uu[r]], R.indptr[uu[r] + 1]
            rows.append(np.concatenate([R.indices[a:b], [hi[r]]]))
        sc = {name: REG.full_scores(engines[name], uu) for name in ENGINES}
        ar = np.arange(m)
        posn = {name: sc[name][ar, hi] for name in ENGINES}
        for si, s in enumerate(SEEDS):
            neg = [CP.build_negative_pool(perms[s], rows[r], int(hi[r]))
                   for r in range(m)]
            cnts = np.array([len(x) for x in neg], dtype=np.int32)
            maxc = int(cnts.max())
            NEGIDX = np.zeros((m, maxc), dtype=np.int32)
            for r, x in enumerate(neg):
                NEGIDX[r, :len(x)] = x
            idx_ar = ar[:, None]
            for gi, g in enumerate(grid):
                kk = maxc if g is None else min(g - 1, maxc)
                cc = 1 + np.minimum(cnts, kk)
                cand_count[si, gi, s0:e0] = cc
                CAND = np.concatenate([hi[:, None].astype(np.int32),
                                       NEGIDX[:, :kk]], axis=1)
                need_mask = bool((cnts < kk).any())
                if need_mask:
                    valid = np.concatenate(
                        [np.ones((m, 1), dtype=bool),
                         (np.arange(kk)[None, :] < cnts[:, None])], axis=1)
                for name in ENGINES:
                    cand_sc = sc[name][idx_ar, CAND]
                    if need_mask:
                        cand_sc = np.where(valid, cand_sc, -np.inf)
                    rank = (cand_sc > posn[name][:, None]).sum(1).astype(np.int32)
                    rank_nr[name][si, gi, s0:e0] = rank
                    del cand_sc
                del CAND
            del NEGIDX, neg
        for name in ENGINES:
            del sc[name]
        del sc
        log(f"  chunk {s0:>6}-{e0:<6} done (elapsed {time.time()-t0:.0f}s)")

    # ---- bit-level self-check: if results/ranks.npz exists with the same shape, ICF must match bit-for-bit
    stored = os.path.join(C.OUT_DIR, "ranks.npz")
    if os.path.exists(stored):
        zs = np.load(stored, allow_pickle=True)
        for e in ENGINES:
            key = f"rank_nr_{e}"
            if key not in zs.files:
                continue
            sr = zs[key]
            if sr.shape != rank_nr[e].shape:
                log(f"  [{e}] stored shape {sr.shape} != current {rank_nr[e].shape} (cohort size differs, skipping bit-level check)")
                continue
            same = np.array_equal(sr, rank_nr[e])
            log(f"  [{e}] bit-identical to stored ranks: {same}")
            if e == "ICF" and not same:
                raise SystemExit("!! ICF ranks must be bit-identical; a mismatch means the "
                                 "reproduction environment changed. Stopping.")

    # ---- per-tier winner (no-rerank HR@5 criterion; SVD never wins)
    winner = {}
    for gi, gl in enumerate(grid_label):
        vals = {e: float((rank_nr[e][:, gi, :] < C.HR_K).mean()) for e in ENGINES}
        winner[gi] = max(("ICF", "NCF"), key=lambda e: vals[e])
        log(f"  pool {gl:>5}: winner {winner[gi]}   "
            + "  ".join(f"{e}={vals[e]:.5f}" for e in ENGINES))

    # ------------------------------------- phase B: GA rerank + list-level metrics
    log("\n[4/5] phase B: running four rerank arms on the winner engine (greedy/randw/randp/GA) ...")
    t1 = time.time()
    for s0 in range(0, n, args.chunk):
        e0 = min(s0 + args.chunk, n)
        uu = cohort_u[s0:e0]
        hi = hidden_i[s0:e0]
        m = e0 - s0
        rows = []
        for r in range(m):
            a, b = R.indptr[uu[r]], R.indptr[uu[r] + 1]
            rows.append(np.concatenate([R.indices[a:b], [hi[r]]]))
        sc = {name: REG.full_scores(engines[name], uu) for name in ENGINES}
        ar = np.arange(m)
        for si, s in enumerate(SEEDS):
            neg = [CP.build_negative_pool(perms[s], rows[r], int(hi[r]))
                   for r in range(m)]
            cnts = np.array([len(x) for x in neg], dtype=np.int32)
            maxc = int(cnts.max())
            NEGIDX = np.zeros((m, maxc), dtype=np.int32)
            for r, x in enumerate(neg):
                NEGIDX[r, :len(x)] = x
            idx_ar = ar[:, None]
            for gi, g in enumerate(grid):
                w = winner[gi]
                gk = CP.FULL_GRID_KEY if g is None else int(g)   # RNG stream uses pool size, not grid index
                kk = maxc if g is None else min(g - 1, maxc)
                cc = 1 + np.minimum(cnts, kk)
                CAND = np.concatenate([hi[:, None].astype(np.int32),
                                       NEGIDX[:, :kk]], axis=1)
                need_mask = bool((cnts < kk).any())
                if need_mask:
                    valid = np.concatenate(
                        [np.ones((m, 1), dtype=bool),
                         (np.arange(kk)[None, :] < cnts[:, None])], axis=1)
                cand_sc = sc[w][idx_ar, CAND]
                if need_mask:
                    cand_sc = np.where(valid, cand_sc, -np.inf)
                depth_arr = np.minimum(cc, CP.RERANK_DEPTH)
                gm = {k: np.zeros(m, np.float32) for k in GA_METRICS}
                rm = {k: np.zeros(m, np.float32) for k in GREEDY_METRICS}
                wm = {k: np.zeros(m, np.float32) for k in RANDOM_METRICS}
                pm = {k: np.zeros(m, np.float32) for k in RANDOM_METRICS}
                for r in range(m):
                    h = int(hi[r])
                    d = int(depth_arr[r])
                    uid = int(uu[r])
                    s_r = cand_sc[r, :cc[r]]
                    # descending-relevance window (identical selection to the main experiment)
                    if d <= C.HR_K:
                        sel = np.argsort(-s_r, kind="stable")[:d]
                    else:
                        sel = np.argpartition(-s_r, d - 1)[:d]
                        sel = sel[np.argsort(-s_r[sel], kind="stable")]
                    p_i = CAND[r, sel]
                    p_s = s_r[sel]
                    assert np.all(np.diff(p_s.astype(np.float64)) <= 0), \
                        "rerank window must already be sorted by relevance descending"
                    # ---------------- four rerank arms: same pool, only the selection rule differs
                    # (1) GA
                    if d <= C.HR_K:
                        ga_items = list(p_i)
                    else:
                        items, _, _ = rerank_ga(
                            p_i, p_s, tag_sets, novelty, fresh,
                            np.zeros(len(p_i), dtype=np.float32),
                            CP.user_rng(s, gk, w, uid))
                        ga_items = list(items)
                    # (2) greedy (pure-relevance Top-10, = no-rerank Top-10)
                    greedy_items = list(rerank_greedy(p_i, p_s)[0])
                    # (3) randw: pick 10 at random within the top-20 by relevance
                    win_items = list(rerank_random(
                        p_i, p_s,
                        CP.arm_rng(s, gk, w, uid, CP.ARM_ID["rand_window"]))[0])
                    # (4) randp: pick 10 at random from the whole candidate pool
                    pool_items = list(rerank_random(
                        CAND[r, :cc[r]], s_r,
                        CP.arm_rng(s, gk, w, uid, CP.ARM_ID["rand_pool"]))[0])
                    # one shared metric implementation, applied once per arm
                    for arm_items, dst in ((ga_items, gm), (greedy_items, rm),
                                           (win_items, wm), (pool_items, pm)):
                        vals = _arm_metrics(arm_items, tag_sets, novelty, fresh,
                                            nov_thr, h)
                        for k in dst:
                            dst[k][r] = vals[k]
                for k, vv in gm.items():
                    ga_m[w][k][si, gi, s0:e0] = vv
                for k, vv in rm.items():
                    gr_m[w][k][si, gi, s0:e0] = vv
                for k, vv in wm.items():
                    randw_m[k][si, gi, s0:e0] = vv
                for k, vv in pm.items():
                    randp_m[k][si, gi, s0:e0] = vv
                del cand_sc, CAND
            del NEGIDX, neg
        for name in ENGINES:
            del sc[name]
        del sc
        log(f"  chunk {s0:>6}-{e0:<6} done (elapsed {time.time()-t1:.0f}s)")

    # ------------------------------------------------------------ aggregate
    log("\n[5/5] Aggregating ...")
    tables = {}
    for gi, gl in enumerate(grid_label):
        w = winner[gi]
        cc = cand_count[:, gi, :].astype(np.float64)
        rand5 = float((np.minimum(C.HR_K, cc) / np.maximum(cc, 1.0)).mean())
        rand10 = float((np.minimum(10, cc) / np.maximum(cc, 1.0)).mean())
        nr5 = float((rank_nr[w][:, gi, :] < C.HR_K).mean())
        nr10 = float((rank_nr[w][:, gi, :] < 10).mean())
        ga5 = float(ga_m[w]["hr5"][:, gi, :].mean())
        ga10 = float(ga_m[w]["hr10"][:, gi, :].mean())
        # random control arms (randw / randp)
        rw5 = float(randw_m["hr5"][:, gi, :].mean())
        rw10 = float(randw_m["hr10"][:, gi, :].mean())
        rp5 = float(randp_m["hr5"][:, gi, :].mean())
        rp10 = float(randp_m["hr10"][:, gi, :].mean())
        row = dict(
            pool=gl, engine=w, mean_candidate_set=round(float(cc.mean()), 1),
            random_hr5=rand5, random_hr10=rand10,
            nr_hr5=round(nr5, 5), nr_hr10=round(nr10, 5),
            nr_lift5=round(nr5 / rand5, 2), nr_lift10=round(nr10 / rand10, 2),
            ga_hr5=round(ga5, 5), ga_hr10=round(ga10, 5),
            ga_dpp5=round((ga5 - nr5) * 100, 3),
            ga_dpp10=round((ga10 - nr10) * 100, 3),
            ga_lift5=round(ga5 / rand5, 2), ga_lift10=round(ga10 / rand10, 2),
            ga_nov5=round(float(ga_m[w]["nov5"][:, gi, :].mean()), 4),
            ga_nov10=round(float(ga_m[w]["nov10"][:, gi, :].mean()), 4),
            ga_div10=round(float(ga_m[w]["div10"][:, gi, :].mean()), 4),
            ga_fresh10=round(float(ga_m[w]["fresh10"][:, gi, :].mean()), 4),
            ga_lt10=round(float(ga_m[w]["lt10"][:, gi, :].mean()), 4),
            greedy_nov5=round(float(gr_m[w]["nov5"][:, gi, :].mean()), 4),
            greedy_nov10=round(float(gr_m[w]["nov10"][:, gi, :].mean()), 4),
            greedy_div10=round(float(gr_m[w]["div10"][:, gi, :].mean()), 4),
            greedy_fresh10=round(float(gr_m[w]["fresh10"][:, gi, :].mean()), 4),
            greedy_lt10=round(float(gr_m[w]["lt10"][:, gi, :].mean()), 4),
            randw_hr5=round(rw5, 5), randw_hr10=round(rw10, 5),
            randw_nov5=round(float(randw_m["nov5"][:, gi, :].mean()), 4),
            randw_nov10=round(float(randw_m["nov10"][:, gi, :].mean()), 4),
            randw_div10=round(float(randw_m["div10"][:, gi, :].mean()), 4),
            randw_fresh10=round(float(randw_m["fresh10"][:, gi, :].mean()), 4),
            randw_lt10=round(float(randw_m["lt10"][:, gi, :].mean()), 4),
            randp_hr5=round(rp5, 5), randp_hr10=round(rp10, 5),
            randp_nov5=round(float(randp_m["nov5"][:, gi, :].mean()), 4),
            randp_nov10=round(float(randp_m["nov10"][:, gi, :].mean()), 4),
            randp_div10=round(float(randp_m["div10"][:, gi, :].mean()), 4),
            randp_fresh10=round(float(randp_m["fresh10"][:, gi, :].mean()), 4),
            randp_lt10=round(float(randp_m["lt10"][:, gi, :].mean()), 4),
            # self-check: measured randp hit rate should equal the analytic value K/N
            randp_hr10_gap=round(abs(rp10 - rand10), 8),
        )
        row["d_nov10"] = round(row["ga_nov10"] - row["greedy_nov10"], 4)
        row["d_nov5"] = round(row["ga_nov5"] - row["greedy_nov5"], 4)
        row["d_lt10"] = round(row["ga_lt10"] - row["greedy_lt10"], 4)
        row["d_div10"] = round(row["ga_div10"] - row["greedy_div10"], 4)
        # GA's reranking ability = net gain over the WINDOW-RANDOM arm (randomness itself removed)
        row["ga_over_randw_nov10"] = round(row["ga_nov10"] - row["randw_nov10"], 4)
        row["ga_over_randw_lt10"] = round(row["ga_lt10"] - row["randw_lt10"], 4)
        row["ga_over_randw_hr10"] = round(row["ga_hr10"] - row["randw_hr10"], 5)
        tables[gl] = row
        log(f"  {gl:>5} [{w}]  HR@10  greedy={nr10:.4f} randw={rw10:.4f} randp={rp10:.4f} GA={ga10:.4f}"
            f"   long-tail share  greedy={row['greedy_lt10']:.4f} randw={row['randw_lt10']:.4f} "
            f"randp={row['randp_lt10']:.4f} GA={row['ga_lt10']:.4f}")
        log(f"          randp self-check |measured HR@10 - analytic K/N| = {row['randp_hr10_gap']:.2e}")

    engines_nr = {}
    for e in ENGINES:
        engines_nr[e] = dict(
            hr5=[round(float((rank_nr[e][:, gi, :] < C.HR_K).mean()), 5) for gi in range(n_grid)],
            hr10=[round(float((rank_nr[e][:, gi, :] < 10).mean()), 5) for gi in range(n_grid)],
        )

    out = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        purpose="ICF + GA main experiment: recall comparison and GA rerank gains on a 10-tier candidate pool",
        protocol=dict(
            users=int(n), seeds=SEEDS,
            sampling="prefix (identical to main run)",
            rerank_depth=CP.RERANK_DEPTH, top_k_out=C.TOP_K_OUT,
            rerank_arms=["greedy: pure-relevance Top-10 (identical to no rerank)",
                         "randw: pick 10 at random within the top-20 by relevance",
                         "randp: pick 10 at random from the whole candidate pool",
                         "ga: genetic algorithm with four-term weighted fitness"],
            ga_rng="default_rng([sampling_seed, pool_size, engine_id, user_id])",
            random_arm_rng=("default_rng([sampling_seed, pool_size, engine_id, user_id, arm])"
                            ", arm=1 randw / arm=2 randp; independent of the GA stream, "
                            "so adding control arms never perturbs any published GA numbers"),
            arm_display_order="all arms first pick 10 items, then display them in descending relevance order, matching the GA output convention",
            ga_fitness="0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 freshness (slide spec)",
            ga_weights=dict(relevance=C.GA_W_RELEVANCE, diversity=C.GA_W_DIVERSITY,
                            novelty=C.GA_W_NOVELTY, fresh=C.GA_W_FRESH),
            ga_rel_norm="relevance term is min-max normalized to [0,1] within the pool before weighting so the four terms are comparable",
            novelty_item="novelty_train = 1 - log(1+popularity)/log(1+max popularity) (higher = more niche)",
            longtail_threshold=round(nov_thr, 4),
            longtail_def="share of the top 10 items with novelty >= the catalog-wide median",
            randp_check="measured randp HR should equal the analytic value K/N (see tables[*].randp_hr10_gap)",
            engines_retrained=["ICF (deterministic, bit-level verified)", "SVD (retrained with fixed seed)", "NCF (retrained with fixed seed)"],
            winner_rule="per-pool no-rerank HR@5 argmax among {ICF, NCF} (SVD never wins on any pool)"),
        tables=tables,
        engines_no_rerank=engines_nr,
        wall_seconds=round(time.time() - t_all, 1),
        environment=dict(python=platform.python_version(),
                         platform=platform.platform()),
    )

    sfx = f"_{args.tag}" if args.tag else ""
    jp = os.path.join(C.OUT_DIR, f"experiment{sfx}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    rp = os.path.join(C.OUT_DIR, f"ranks{sfx}.npz")
    np.savez_compressed(
        rp, grid=np.array([grid_label], dtype=object), seeds=np.array(SEEDS),
        winner_engine=np.array([winner[gi] for gi in range(n_grid)], dtype=object),
        cand_count=cand_count, cohort_u=cohort_u, hidden_i=hidden_i,
        **{f"rank_nr_{e}": v for e, v in rank_nr.items()},
        **{f"ga_{e}_{k}": v for e in ENGINES for k, v in ga_m[e].items()},
        **{f"greedy_{e}_{k}": v for e in ENGINES for k, v in gr_m[e].items()},
        **{f"randw_{k}": v for k, v in randw_m.items()},
        **{f"randp_{k}": v for k, v in randp_m.items()})
    log(f"\n  -> {jp}\n  -> {rp}")
    log(f"\nTotal time {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
