# -*- coding: utf-8 -*-
"""
analyze_icf_part.py — produces all comparison data for "Part 1 - ICF" of the report.

Question answered: across the 10 candidate-pool sizes, did each recall engine
(ICF / SVD / NCF) rank the held-out positive into Top-5 / Top-10, and how much
does that beat pure random guessing?

Data: `engines_no_rerank` and `tables` in results/experiment.json, plus
results/ranks.npz. HR@K is recomputed from per-user ranks; `lift` = HR@K / (K/N)
is the only pool-size-comparable metric. Output: results/icf_part.json.
"""
import json
import os

import numpy as np

import config as C

SRC_NPZ = os.path.join(C.OUT_DIR, "ranks.npz")
SRC_JSON = os.path.join(C.OUT_DIR, "experiment.json")
OUT = os.path.join(C.OUT_DIR, "icf_part.json")

ENGINES = ["ICF", "SVD", "NCF"]


def novelty_stats():
    """Recompute novelty distribution stats from the raw item_features.jsonl.

    Deliberately not read from any cached json; re-derives
    novelty = 1 - log(1+popularity) / log(1+max popularity) and verifies it
    exactly, so the report's novelty-definition section is backed by evidence.
    """
    nov, pop = [], []
    with open(C.F_ITEM_FEATURES, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            nov.append(o["novelty_train"])
            pop.append(o["rating_count_train"])
    nov = np.asarray(nov, dtype=np.float64)
    pop = np.asarray(pop, dtype=np.float64)

    # formula self-check: deviation must be exactly 0 (same float op path)
    chk = 1.0 - np.log1p(pop) / np.log1p(pop.max())
    dev = float(np.max(np.abs(chk - nov)))
    if dev > 1e-12:
        raise SystemExit(f"[fatal] novelty formula check failed, max deviation {dev:.3e}")

    med = float(np.median(nov))
    order = np.argsort(-pop)
    return dict(
        formula="novelty(i) = 1 - log(1 + rating_count_train_i) / log(1 + max(rating_count_train))",
        formula_max_dev=dev,
        n_items=int(nov.size),
        novel_min=float(nov.min()), novel_max=float(nov.max()),
        novel_mean=float(nov.mean()), novel_median=med,
        n_longtail=int((nov >= med).sum()),
        longtail_share=float((nov >= med).mean()),
        n_zero_pop=int((nov >= 1.0 - 1e-9).sum()),
        n_hottest=int((nov <= 1e-9).sum()),
        pop_min=float(pop.min()), pop_max=float(pop.max()), pop_median=float(np.median(pop)),
        head1_share=float(pop[order][: nov.size // 100].sum() / pop.sum()),
        head10_share=float(pop[order][: nov.size // 10].sum() / pop.sum()),
    )



def main():
    exp = json.load(open(SRC_JSON, encoding="utf-8"))
    z = np.load(SRC_NPZ, allow_pickle=True)

    grid = [str(g) for g in z["grid"][0]]
    seeds = [int(s) for s in z["seeds"]]
    n_users = int(z["rank_nr_ICF"].shape[2])

    print("=" * 96)
    print(f"Part 1 - ICF: 3 recall engines x {len(grid)} candidate-pool sizes (no rerank)")
    print(f"users {n_users:,}   seeds {seeds}")
    print("=" * 96)
    hdr = f"{'pool N':>7} {'cand set':>9} {'win':>4} | " + " | ".join(
        f"{e:>22}" for e in ENGINES)
    print(hdr)
    print(f"{'':>7} {'':>9} {'':>4} | " + " | ".join(
        f"{'HR@5':>9}{'HR@10':>12}" for _ in ENGINES))
    print("-" * 96)

    per_pool = {}
    for gl in grid:
        t = exp["tables"][gl]
        nr = exp["engines_no_rerank"]
        pool_rec = dict(
            mean_candidate_set=t["mean_candidate_set"],
            random_hr5=t["random_hr5"], random_hr10=t["random_hr10"],
            winner=t["engine"], engines={},
        )
        for e in ENGINES:
            # recompute from per-user ranks, not the rounded experiment.json values
            gi = grid.index(gl)
            r = z[f"rank_nr_{e}"][:, gi, :]
            hr5 = float((r < C.HR_K).mean())
            hr10 = float((r < 10).mean())
            pool_rec["engines"][e] = dict(
                hr5=hr5, hr10=hr10,
                lift5=hr5 / t["random_hr5"] if t["random_hr5"] else float("nan"),
                lift10=hr10 / t["random_hr10"] if t["random_hr10"] else float("nan"),
            )
        per_pool[gl] = pool_rec
        cells = " | ".join(
            f"{pool_rec['engines'][e]['hr5']*100:8.2f}% {pool_rec['engines'][e]['hr10']*100:10.2f}%"
            for e in ENGINES)
        print(f"{gl:>7} {t['mean_candidate_set']:>9.1f} {t['engine']:>4} | {cells}")

    # lift table (pool-size-comparable metric)
    print()
    print(f"{'pool N':>7} | " + " | ".join(f"{e:>13}" for e in ENGINES))
    print(f"{'':>7} | " + " | ".join(f"{'@5':>6}{'@10':>7}" for _ in ENGINES))
    print("-" * 70)
    for gl in grid:
        p = per_pool[gl]
        cells = " | ".join(
            f"{p['engines'][e]['lift5']:5.1f}×{p['engines'][e]['lift10']:6.1f}×"
            for e in ENGINES)
        print(f"{gl:>7} | {cells}")

    lift5 = [per_pool[g]["engines"][e]["lift5"] for g in grid for e in ENGINES]
    lift10 = [per_pool[g]["engines"][e]["lift10"] for g in grid for e in ENGINES]
    wins = {e: sum(1 for g in grid if per_pool[g]["winner"] == e) for e in ENGINES}

    nv = novelty_stats()
    print()
    print("Novelty distribution (basis for the long-tail definition)")
    print(f"  formula check max deviation = {nv['formula_max_dev']:.1e}  -> passed")
    print(f"  items {nv['n_items']:,}   novelty mean {nv['novel_mean']:.4f} "
          f"median {nv['novel_median']:.4f}   long-tail threshold = median = {nv['novel_median']:.4f}")
    print(f"  long-tail items {nv['n_longtail']:,} ({nv['longtail_share']*100:.1f}%)"
          f"   zero-popularity {nv['n_zero_pop']}   hottest {nv['n_hottest']}")
    print(f"  head concentration: top 1% of items hold {nv['head1_share']*100:.1f}% of popularity, "
          f"top 10% hold {nv['head10_share']*100:.1f}%")

    out = dict(
        source="results/ranks.npz + results/experiment.json + data/ga/item_features.jsonl",
        part="Part 1 - ICF (no-rerank recall comparison)",
        n_users=n_users, seeds=seeds,
        pools=grid, engines=ENGINES,
        per_pool=per_pool,
        novelty=nv,
        summary=dict(
            lift5_range=[float(min(lift5)), float(max(lift5))],
            lift10_range=[float(min(lift10)), float(max(lift10))],
            winner_count=wins,
            full=dict(
                pool="full",
                per_engine={e: dict(hr5=per_pool["full"]["engines"][e]["hr5"],
                                    hr10=per_pool["full"]["engines"][e]["hr10"],
                                    lift5=per_pool["full"]["engines"][e]["lift5"],
                                    lift10=per_pool["full"]["engines"][e]["lift10"])
                            for e in ENGINES},
                random_hr5=per_pool["full"]["random_hr5"],
                random_hr10=per_pool["full"]["random_hr10"],
                n_candidates=per_pool["full"]["mean_candidate_set"],
            ),
        ),
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
