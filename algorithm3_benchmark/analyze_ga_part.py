# -*- coding: utf-8 -*-
"""
analyze_ga_part.py — produces all comparison data for report part 2 (rerank contrast).

Four rerank arms over the same candidate pool and recall result: greedy
(pure-relevance Top-10, identical to no rerank), randw (random within the top-20
by relevance), randp (random from the whole pool), and GA (0.4 relevance + 0.3
diversity + 0.2 novelty + 0.1 freshness).

Beyond nov@10, gains are expressed in three tangible metrics:
  GOS@10  GA objective score = 0.3*div@10 + 0.2*nov@10 + 0.1*fresh@10
  LTE@10  long-tail exposure = sum over users of (lt10 * 10)
  LTR@10  long-tail touch rate = share of users with >=1 long-tail item in top 10
Cost is monitored with HR@10 / HR@5.

Paired protocol: per-user diffs under the same user / pool / engine / seed.
Wilcoxon signed-rank for continuous metrics, McNemar for binary hit rates;
95% CIs and Cohen's d_z are reported.

Input:  results/ranks.npz
Output: results/ga_part.json
"""
import json
import os
from math import erfc, sqrt

import numpy as np

import config as C

NPZ = os.path.join(C.OUT_DIR, "ranks.npz")
OUT = os.path.join(C.OUT_DIR, "ga_part.json")

CONT = ("nov5", "nov10", "div10", "fresh10", "lt10")
ARMS = ("greedy", "randw", "randp", "ga")
GENERATIVE = ("randw", "randp")

W_REL = C.GA_W_RELEVANCE
W_DIV = C.GA_W_DIVERSITY
W_NOV = C.GA_W_NOVELTY
W_FRESH = C.GA_W_FRESH

TOP_K = 10
LONGTAIL_MEDIAN = 0.6156   # catalog-wide novelty median (from item_features)

ARM_DEF = {
    "greedy": "greedy: pure-relevance Top-10 (identical to no rerank)",
    "randw": "randw: pick 10 at random within the top-20 by relevance",
    "randp": "randp: pick 10 at random from the whole candidate pool",
    "ga": "GA: 0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 freshness",
}


# ------------------------------------------------------------------ stats tools
def paired_stats(a, b):
    d = a - b
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / sqrt(n) if n else 0.0
    return dict(
        n=int(n), mean=mean, sd=sd,
        ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se,
        up=int((d > 1e-9).sum()), flat=int((np.abs(d) <= 1e-9).sum()),
        down=int((d < -1e-9).sum()),
        up_pct=100.0 * (d > 1e-9).sum() / n,
        flat_pct=100.0 * (np.abs(d) <= 1e-9).sum() / n,
        down_pct=100.0 * (d < -1e-9).sum() / n,
        dz=mean / sd if sd > 0 else 0.0,
        n_nonzero=int((d != 0).sum()),
    )


def wilcoxon_p(d):
    nz = d[d != 0]
    if nz.size < 25:
        return float("nan")
    r = np.argsort(np.argsort(np.abs(nz))) + 1.0
    t_plus = float(r[nz > 0].sum())
    n = nz.size
    mu = n * (n + 1) / 4.0
    sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (t_plus - mu) / sd if sd > 0 else 0.0
    return float(erfc(abs(z) / sqrt(2.0)))


def mcnemar(a, b):
    """a = new-arm hits, b = control-arm hits (per-user 0/1). Returns (p, ref-wrong/new-right, ref-right/new-wrong)."""
    a = a > 0.5
    b = b > 0.5
    n01 = int((~b & a).sum())
    n10 = int((b & ~a).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    z = (abs(n01 - n10) - 1.0) / sqrt(n)
    return float(erfc(abs(z) / sqrt(2.0))), n01, n10


# ------------------------------------------------------------ unified data access for the four arms
def mean_series(z, gi, w, arm, key):
    """Per-user series for one arm on one tier, averaged across seeds.

    The order of mean(0) and astype is kept verbatim from the historical version
    (non-HR keys: mean first then astype; HR keys: astype first then mean) so the
    greedy / ga columns reproduce the published report bit-for-bit.
    """
    if arm == "greedy":
        if key in ("hr5", "hr10"):
            k = 5 if key == "hr5" else 10
            return (z[f"rank_nr_{w}"][:, gi, :] < k).astype(np.float64).mean(0)
        return z[f"greedy_{w}_{key}"][:, gi, :].mean(0).astype(np.float64)
    if arm == "ga":
        v = z[f"ga_{w}_{key}"][:, gi, :]
    else:
        v = z[f"{arm}_{key}"][:, gi, :]
    if key in ("hr5", "hr10"):
        return (v > 0.5).astype(np.float64).mean(0)
    return v.mean(0).astype(np.float64)


def ltr_seed(z, gi, w, arm):
    """Per-seed "touched >=1 long-tail item" flags, shape (n_seeds, N).

    Must threshold per seed (>0) BEFORE averaging across seeds; averaging first
    would count users exposed under only one seed as touched and inflate LTR.
    """
    key = {"greedy": f"greedy_{w}_lt10", "ga": f"ga_{w}_lt10",
           "randw": "randw_lt10", "randp": "randp_lt10"}[arm]
    return z[key][:, gi, :] > 0


def user_novelty_mean(z):
    """Mean novelty of users' truly consumed items, represented by the hidden item."""
    nov = {}
    with open(C.F_ITEM_FEATURES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            nov[i] = json.loads(line)["novelty_train"]
    hidden = z["hidden_i"]
    vals = np.asarray([nov[int(i)] for i in hidden], dtype=np.float64)
    med = float(np.median(list(nov.values())))
    return dict(
        mean=float(vals.mean()), median=float(np.median(vals)),
        longtail_share=float((vals >= med).mean()),
        n_users=int(vals.size), threshold=med,
    )



def arm_block(z, gi, w, arm, n_users):
    """Benefit / cost snapshot of one rerank arm on one tier."""
    S = {k: mean_series(z, gi, w, arm, k)
         for k in ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")}
    div, nov, fre, lt = S["div10"], S["nov10"], S["fresh10"], S["lt10"]
    ltseed = ltr_seed(z, gi, w, arm)

    gos = W_DIV * div + W_NOV * nov + W_FRESH * fre
    lte = float(lt.sum() * TOP_K)
    ltr = float(ltseed.mean(0).mean())

    return dict(
        arm=arm, label=ARM_DEF[arm], series=S,
        lt=lt, lt_seed=ltseed, gos=gos,
        lte=lte, ltr=ltr, lte_per_user=float(lt.sum()),
        gos_mean=float(gos.mean()),
        hr5_mean=float(S["hr5"].mean()), hr10_mean=float(S["hr10"].mean()),
        nov5_mean=float(S["nov5"].mean()), nov10_mean=float(nov.mean()),
        div10_mean=float(div.mean()), fresh10_mean=float(fre.mean()),
        lt10_mean=float(lt.mean()),
    )


def pair(a, b, n_users):
    """Benefit / cost contrast of new arm a vs reference arm b (per-user paired)."""
    return dict(
        new=a["arm"], ref=b["arm"],
        lte=dict(ref=b["lte"], new=a["lte"], delta=a["lte"] - b["lte"],
                 rel_pct=100.0 * (a["lte"] - b["lte"]) / b["lte"] if b["lte"] else float("nan")),
        ltr=dict(ref=b["ltr"], new=a["ltr"],
                 ref_users=b["ltr"] * n_users, new_users=a["ltr"] * n_users,
                 delta_users=(a["ltr"] - b["ltr"]) * n_users,
                 rel_pct=100.0 * (a["ltr"] - b["ltr"]) / b["ltr"] if b["ltr"] else float("nan")),
        gos=dict(ref=b["gos_mean"], new=a["gos_mean"],
                 delta=a["gos_mean"] - b["gos_mean"],
                 rel_pct=100.0 * (a["gos_mean"] - b["gos_mean"]) / b["gos_mean"]
                 if b["gos_mean"] else float("nan"),
                 p=wilcoxon_p(a["gos"] - b["gos"]),
                 dz=float((a["gos"] - b["gos"]).mean() / (a["gos"] - b["gos"]).std(ddof=1))
                 if (a["gos"] - b["gos"]).std(ddof=1) > 0 else 0.0,
                 st=paired_stats(a["gos"], b["gos"])),
        lt10=dict(ref=b["lt10_mean"], new=a["lt10_mean"],
                  delta=a["lt10_mean"] - b["lt10_mean"],
                  rel_pct=100.0 * (a["lt10_mean"] - b["lt10_mean"]) / b["lt10_mean"]
                  if b["lt10_mean"] else float("nan")),
        nov10=dict(ref=b["nov10_mean"], new=a["nov10_mean"],
                   delta=a["nov10_mean"] - b["nov10_mean"],
                   rel_pct=100.0 * (a["nov10_mean"] - b["nov10_mean"]) / b["nov10_mean"]
                   if b["nov10_mean"] else float("nan")),
        hr10=dict(ref=b["hr10_mean"], new=a["hr10_mean"],
                  delta_pp=100.0 * (a["hr10_mean"] - b["hr10_mean"]),
                  rel_pct=100.0 * (a["hr10_mean"] - b["hr10_mean"]) / b["hr10_mean"]
                  if b["hr10_mean"] else float("nan"),
                  **dict(zip(("p", "n_ref_only", "n_new_only"),
                             mcnemar(a["series"]["hr10"], b["series"]["hr10"])))),
        hr5=dict(ref=b["hr5_mean"], new=a["hr5_mean"],
                 delta_pp=100.0 * (a["hr5_mean"] - b["hr5_mean"]),
                 rel_pct=100.0 * (a["hr5_mean"] - b["hr5_mean"]) / b["hr5_mean"]
                 if b["hr5_mean"] else float("nan"),
                 **dict(zip(("p", "n_ref_only", "n_new_only"),
                            mcnemar(a["series"]["hr5"], b["series"]["hr5"])))),
    )


def main():
    z = np.load(NPZ, allow_pickle=True)
    grid = [str(g) for g in z["grid"][0]]
    seeds = [int(s) for s in z["seeds"]]
    N = int(z["cand_count"].shape[2])
    print("=" * 118)
    print("Part 2 - rerank contrast: greedy / randw / randp / GA")
    print(f"seeds {seeds}   users {N:,}   candidate pool tiers {grid}")
    print("=" * 118)

    # per-tier winner engine: no-rerank HR@5 argmax among {ICF, NCF} (same rule as the main experiment)
    winner = []
    for gi, gl in enumerate(grid):
        v = {e: float((z[f"rank_nr_{e}"][:, gi, :] < C.HR_K).mean()) for e in ("ICF", "NCF")}
        winner.append(max(("ICF", "NCF"), key=lambda e: v[e]))

    out = dict(
        source=os.path.basename(NPZ), seeds=seeds, n_users=N,
        winner_rule="argmax no-rerank HR@5 per pool among {ICF, NCF}",
        weights=dict(relevance=W_REL, diversity=W_DIV, novelty=W_NOV, fresh=W_FRESH),
        longtail_median=LONGTAIL_MEDIAN, top_k=TOP_K,
        exposure_slots=N * TOP_K,
        arm_definitions=ARM_DEF,
        nov_user=user_novelty_mean(z),
        pools={}, summary={},
    )
    nvu = out["nov_user"]
    print(f"[novelty] long-tail threshold (catalog median)={nvu['threshold']:.4f}   "
          f"mean novelty of users' consumed items={nvu['mean']:.4f} (median {nvu['median']:.4f}, "
          f"{nvu['longtail_share']*100:.1f}% long-tail)")
    print()

    # self-check 1: greedy HR@10 == no-rerank rank < 10
    gi_full = grid.index("full")
    chk = float((z["rank_nr_ICF"][:, gi_full, :] < 10).mean())
    print(f"[check 1] full-tier ICF greedy HR@10 (recomputed from ranks) = {chk:.5f}"
          f"  -> {'PASS' if abs(chk - 0.20925) < 1e-4 else 'MISMATCH vs historical value 0.20925!'}")
    # self-check 2: measured randp hit rate should equal the analytic K/N ("random guess" level)
    ccf = z["cand_count"][:, gi_full, :].astype(np.float64)
    ana10 = float((np.minimum(10, ccf) / np.maximum(ccf, 1.0)).mean())
    mea10 = float((z["randp_hr10"][:, gi_full, :] > 0.5).astype(np.float64).mean())
    print(f"[check 2] full-tier randp HR@10 measured {mea10:.6f} vs analytic K/N {ana10:.6f}"
          f"  -> {'PASS' if abs(mea10 - ana10) < 2e-4 else 'MISMATCH!'}")
    print()

    hdr = (f"{'pool N':>7} {'eng':>4} | {'HR@10':>6} {'greedy':>7} {'randw':>7} {'randp':>7} {'GA':>7} | "
           f"{'lt10 share':>13} {'greedy':>7} {'randw':>7} {'randp':>7} {'GA':>7} | "
           f"{'LTE@10 greedy':>13} {'GA':>8}")
    print(hdr)
    print("-" * 118)

    rows = {}
    for gi, gl in enumerate(grid):
        w = winner[gi]
        A = {a: arm_block(z, gi, w, a, N) for a in ARMS}
        g, rw, rp, gaa = A["greedy"], A["randw"], A["randp"], A["ga"]
        rec = dict(pool=gl, engine=w,
                   n_candidates=float(z["cand_count"][:, gi, :].mean()))

        # randp self-check: measured HR@10 must equal the analytic min(K,N)/N
        ccf = z["cand_count"][:, gi, :].astype(np.float64)
        rec["randp_check"] = dict(
            analytic=float((np.minimum(TOP_K, ccf) / np.maximum(ccf, 1.0)).mean()),
            measured=float((z["randp_hr10"][:, gi, :] > 0.5).astype(np.float64).mean()),
        )

        # four-arm snapshot (the report main table reads from here)
        rec["arms"] = {a: dict(
            label=A[a]["label"], lte=A[a]["lte"], ltr=A[a]["ltr"],
            ltr_users=A[a]["ltr"] * N, lte_per_user=A[a]["lte_per_user"],
            gos=A[a]["gos_mean"], hr5=A[a]["hr5_mean"], hr10=A[a]["hr10_mean"],
            nov5=A[a]["nov5_mean"], nov10=A[a]["nov10_mean"],
            div10=A[a]["div10_mean"], fresh10=A[a]["fresh10_mean"],
            lt10=A[a]["lt10_mean"],
        ) for a in ARMS}

        # GA vs greedy (historical spec, bit-identical to the published report)
        gos_g, gos_a = g["gos"], gaa["gos"]
        ltr_g_seed, ltr_a_seed = g["lt_seed"], gaa["lt_seed"]
        rec["benefit"] = dict(
            lte=dict(greedy=g["lte"], ga=gaa["lte"], delta=gaa["lte"] - g["lte"],
                     rel_pct=100.0 * (gaa["lte"] - g["lte"]) / g["lte"] if g["lte"] else float("nan")),
            ltr=dict(greedy=g["ltr"], ga=gaa["ltr"],
                     greedy_users=g["ltr"] * N, ga_users=gaa["ltr"] * N,
                     delta_users=(gaa["ltr"] - g["ltr"]) * N,
                     rel_pct=100.0 * (gaa["ltr"] - g["ltr"]) / g["ltr"] if g["ltr"] else float("nan")),
            gos=dict(greedy=float(gos_g.mean()), ga=float(gos_a.mean()),
                     mean=float((gos_a - gos_g).mean()),
                     ci_lo=float((gos_a - gos_g).mean() - 1.96 * (gos_a - gos_g).std(ddof=1) / sqrt(N)),
                     ci_hi=float((gos_a - gos_g).mean() + 1.96 * (gos_a - gos_g).std(ddof=1) / sqrt(N)),
                     p=wilcoxon_p(gos_a - gos_g),
                     dz=float((gos_a - gos_g).mean() / (gos_a - gos_g).std(ddof=1)),
                     up_pct=100.0 * ((gos_a - gos_g) > 1e-9).mean(),
                     flat_pct=100.0 * (np.abs(gos_a - gos_g) <= 1e-9).mean(),
                     down_pct=100.0 * ((gos_a - gos_g) < -1e-9).mean(),
                     rel_pct=100.0 * (gos_a.mean() - gos_g.mean()) / gos_g.mean()),
            newly_exposed_users=float(((ltr_a_seed & ~ltr_g_seed).mean(0)).mean()) * N,
            lost_exposure_users=float(((~ltr_a_seed & ltr_g_seed).mean(0)).mean()) * N,
            lt_mult=float(gaa["lte"] / g["lte"]) if g["lte"] else float("nan"),
        )

        # cost side: GA vs greedy (HR@5 / HR@10)
        cost = {}
        for key in ("hr5", "hr10"):
            a_s, b_s = gaa["series"][key], g["series"][key]
            st = paired_stats(a_s, b_s)
            st["greedy_mean"] = float(b_s.mean())
            st["ga_mean"] = float(a_s.mean())
            st["rel_pct"] = (100.0 * (st["ga_mean"] - st["greedy_mean"]) / st["greedy_mean"]
                             if st["greedy_mean"] else float("nan"))
            p, n01, n10 = mcnemar(a_s, b_s)
            st.update(p=p, test="McNemar", n_greedy_only=n01, n_ga_only=n10)
            cost[key] = st
        rec["cost"] = cost

        # per-metric paired tests (GA vs greedy)
        met = {}
        for m in CONT:
            a_s, b_s = gaa["series"][m], g["series"][m]
            st = paired_stats(a_s, b_s)
            st["greedy_mean"] = float(b_s.mean())
            st["ga_mean"] = float(a_s.mean())
            st["rel_pct"] = (100.0 * (st["ga_mean"] - st["greedy_mean"]) / st["greedy_mean"]
                             if st["greedy_mean"] else float("nan"))
            st["p"] = wilcoxon_p(a_s - b_s)
            met[m] = st
        rec["metrics"] = met

        # additional pairs: GA vs randw / GA vs randp / random arms vs greedy
        rec["vs_randw"] = pair(gaa, rw, N)
        rec["vs_randp"] = pair(gaa, rp, N)
        rec["randw_vs_greedy"] = pair(rw, g, N)
        rec["randp_vs_greedy"] = pair(rp, g, N)
        # share of the randp exposure dividend captured by GA (LTE gain basis)
        gain_randp = rp["lte"] - g["lte"]
        rec["share_of_randp_lte"] = float((gaa["lte"] - g["lte"]) / gain_randp) if gain_randp else float("nan")

        # benefit/cost trade-off: % of long-tail exposure gained per 1pp HR@10 lost
        d_hr10_pp = 100.0 * (cost["hr10"]["ga_mean"] - cost["hr10"]["greedy_mean"])
        lte_rel = rec["benefit"]["lte"]["rel_pct"]
        rec["tradeoff"] = dict(
            d_hr10_pp=d_hr10_pp, lte_rel_pct=lte_rel,
            lte_pct_per_hr10_pp=(abs(lte_rel / d_hr10_pp) if d_hr10_pp < 0 else float("nan")),
            lte_per_user_pct=(100.0 * rec["benefit"]["lte"]["delta"] / N),
        )

        rows[gl] = rec
        out["pools"][gl] = rec
        print(f"{gl:>7} {w:>4} | {cost['hr10']['greedy_mean']*100:6.2f} "
              f"{rw['hr10_mean']*100:7.2f} {rp['hr10_mean']*100:7.2f} {gaa['hr10_mean']*100:7.2f} | "
              f"{g['lt10_mean']:13.4f} {rw['lt10_mean']:7.4f} {rp['lt10_mean']:7.4f} "
              f"{gaa['lt10_mean']:7.4f} | "
              f"{g['lte']:11.0f} {gaa['lte']:8.0f}")

    def rel_txt(x):
        """When the base is 0 the relative gain is 0/0; printing nan would pollute the log."""
        return "base=0" if x != x else f"{x:+.0f}%"

    pos = [g for g in grid
           if rows[g]["benefit"]["lte"]["rel_pct"] == rows[g]["benefit"]["lte"]["rel_pct"]
           and rows[g]["benefit"]["lte"]["rel_pct"] > 0]
    print("\n" + "=" * 118)
    print(f"Tiers with LTE@10 gain (GA vs greedy): {len(pos)}/{len(grid)}")
    print("Per-tier LTE gain: " + ", ".join(
        f"{g}:{rel_txt(rows[g]['benefit']['lte']['rel_pct'])}" for g in grid))
    print("Per-tier LTR gain: " + ", ".join(
        f"{g}:{rel_txt(rows[g]['benefit']['ltr']['rel_pct'])}" for g in grid))
    print("Per-tier GOS gain: " + ", ".join(
        f"{g}:{rows[g]['benefit']['gos']['rel_pct']:+.1f}%" for g in grid))
    print("Per-tier dHR@10: " + ", ".join(
        f"{g}:{rows[g]['tradeoff']['d_hr10_pp']:+.2f}pp" for g in grid))

    full = rows["full"]
    fa = full["arms"]
    print("\nFour-arm contrast (full tier)")
    for a in ARMS:
        print(f"  {a:<7} HR@10={fa[a]['hr10']*100:7.3f}%  lt10={fa[a]['lt10']:.4f}  "
              f"LTE={fa[a]['lte']:9.0f}  LTR={fa[a]['ltr']*100:6.2f}%  GOS={fa[a]['gos']:.4f}")
    print(f"  GA vs greedy  : LTE {full['benefit']['lte']['rel_pct']:+.1f}%  "
          f"dHR@10 {full['tradeoff']['d_hr10_pp']:+.2f}pp")
    print(f"  GA vs randw   : LTE {full['vs_randw']['lte']['rel_pct']:+.1f}%  "
          f"dHR@10 {full['vs_randw']['hr10']['delta_pp']:+.2f}pp  "
          f"p(McNemar)={full['vs_randw']['hr10']['p']:.3g}")
    print(f"  GA vs randp   : LTE {full['vs_randp']['lte']['rel_pct']:+.1f}%  "
          f"dHR@10 {full['vs_randp']['hr10']['delta_pp']:+.2f}pp  "
          f"-> GA captured {full['share_of_randp_lte']*100:.1f}% of the randp exposure dividend")

    print("\nrandp self-check (measured HR@10 should equal the analytic min(K,N)/N)")
    worst = 0.0
    for g in grid:
        rc = rows[g]["randp_check"]
        dev = abs(rc["measured"] - rc["analytic"])
        worst = max(worst, dev)
        print(f"  {g:>7} analytic {rc['analytic']:.6f}   measured {rc['measured']:.6f}   "
              f"deviation {dev:.2e}")
    print(f"  -> max deviation across all {len(grid)} tiers: {worst:.2e}"
          f"  {'PASS' if worst < 5e-3 else 'too large, inspect the random-arm implementation!'}")

    print("\nFour-arm full-tier summary (report table 4.1)")
    for a in ARMS:
        print(f"  {a:<7} HR@5={fa[a]['hr5']*100:7.3f}%  HR@10={fa[a]['hr10']*100:7.3f}%  "
              f"LTE={fa[a]['lte']:9.0f}  LTR={fa[a]['ltr']*100:6.2f}%  "
              f"GOS={fa[a]['gos']:.4f}")

    out["summary"] = dict(
        n_pools=len(grid), lte_positive_pools=len(pos),
        lte_rel_range=[min(rows[g]["benefit"]["lte"]["rel_pct"] for g in grid),
                       max(rows[g]["benefit"]["lte"]["rel_pct"] for g in grid)],
        ltr_rel_range=[min(rows[g]["benefit"]["ltr"]["rel_pct"] for g in grid),
                       max(rows[g]["benefit"]["ltr"]["rel_pct"] for g in grid)],
        gos_rel_range=[min(rows[g]["benefit"]["gos"]["rel_pct"] for g in grid),
                       max(rows[g]["benefit"]["gos"]["rel_pct"] for g in grid)],
        hr10_delta_pp_range=[min(rows[g]["tradeoff"]["d_hr10_pp"] for g in grid),
                             max(rows[g]["tradeoff"]["d_hr10_pp"] for g in grid)],
        four_arm_full={a: dict(
            label=fa[a]["label"], hr5=fa[a]["hr5"] * 100, hr10=fa[a]["hr10"] * 100,
            lt10=fa[a]["lt10"], lte=fa[a]["lte"], lte_per_user=fa[a]["lte_per_user"],
            ltr=fa[a]["ltr"] * 100, ltr_users=fa[a]["ltr_users"], gos=fa[a]["gos"],
            nov5=fa[a]["nov5"], nov10=fa[a]["nov10"], div10=fa[a]["div10"],
            fresh10=fa[a]["fresh10"]) for a in ARMS},
        ga_vs_randw=dict(lte_rel=full["vs_randw"]["lte"]["rel_pct"],
                         hr10_dpp=full["vs_randw"]["hr10"]["delta_pp"],
                         hr10_p=full["vs_randw"]["hr10"]["p"],
                         gos_rel=full["vs_randw"]["gos"]["rel_pct"]),
        ga_vs_randp=dict(lte_rel=full["vs_randp"]["lte"]["rel_pct"],
                         hr10_dpp=full["vs_randp"]["hr10"]["delta_pp"],
                         hr10_p=full["vs_randp"]["hr10"]["p"]),
        share_of_randp_lte=full["share_of_randp_lte"],
        full_pool=dict(
            n_candidates=full["n_candidates"],
            lte_g=full["benefit"]["lte"]["greedy"], lte_a=full["benefit"]["lte"]["ga"],
            lte_rel=full["benefit"]["lte"]["rel_pct"],
            ltr_g=full["benefit"]["ltr"]["greedy_users"],
            ltr_a=full["benefit"]["ltr"]["ga_users"],
            ltr_g_pct=full["benefit"]["ltr"]["greedy"] * 100,
            ltr_a_pct=full["benefit"]["ltr"]["ga"] * 100,
            newly=full["benefit"]["newly_exposed_users"],
            gos_g=full["benefit"]["gos"]["greedy"], gos_a=full["benefit"]["gos"]["ga"],
            gos_rel=full["benefit"]["gos"]["rel_pct"],
            hr10_g=full["cost"]["hr10"]["greedy_mean"] * 100,
            hr10_a=full["cost"]["hr10"]["ga_mean"] * 100,
            hr10_d=full["tradeoff"]["d_hr10_pp"],
            hr5_g=full["cost"]["hr5"]["greedy_mean"] * 100,
            hr5_a=full["cost"]["hr5"]["ga_mean"] * 100,
            hr5_d=100.0 * (full["cost"]["hr5"]["ga_mean"] - full["cost"]["hr5"]["greedy_mean"]),
            per_pp=full["tradeoff"]["lte_pct_per_hr10_pp"],
        ),
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
