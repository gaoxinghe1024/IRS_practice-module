# -*- coding: utf-8 -*-
"""
analyze_icf_part.py — 生成报告"第一部分 · ICF"的全部对比数据。

回答的问题
----------
"三个召回引擎（ICF / SVD / NCF）在 10 档候选池上，各自把正例排进了前 5 / 前 10 吗？
 相比完全随机猜强多少？"

数据来源
--------
results/experiment.json 的 `engines_no_rerank`（3 引擎 × 10 档 HR@5/HR@10）
results/experiment.json 的 `tables`（每档的随机基线、候选集均值、胜者引擎）

读法
----
* HR@5 / HR@10 是"正例进入 Top-5 / Top-10 的用户比例"。候选池越小，
  正例的竞争对手越是随机抽出来的冷门条目，数值自然越高——
  所以**跨档位不能直接比大小，必须连同池规模一起读**。
* 完全随机基线 = K / N。`lift` = HR@K ÷ (K/N)，即"比瞎猜强多少倍"。
  这个倍数才是跨档位可比的口径。

输出
----
results/icf_part.json
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
    """从数据包直接复算新颖度的分布特征（长尾定义的可核查依据）。

    这里刻意**不读缓存的 json**、而是回到原始 item_features.jsonl 重算，
    并在返回前校验 novelty = 1 − log(1+人气) / log(1+最大人气) 精确成立，
    否则报告里"新颖度定义"那一节就只是一段没有证据的说明文字。
    """
    nov, pop = [], []
    with open(C.F_ITEM_FEATURES, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            nov.append(o["novelty_train"])
            pop.append(o["rating_count_train"])
    nov = np.asarray(nov, dtype=np.float64)
    pop = np.asarray(pop, dtype=np.float64)

    # 公式自检：偏差必须精确为 0（同一浮点运算路径）
    chk = 1.0 - np.log1p(pop) / np.log1p(pop.max())
    dev = float(np.max(np.abs(chk - nov)))
    if dev > 1e-12:
        raise SystemExit(f"[致命] 新颖度公式校验失败，最大偏差 {dev:.3e}")

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
    print(f"第一部分 · ICF：3 个召回引擎 × {len(grid)} 档候选池（无重排）")
    print(f"用户 {n_users:,}   种子 {seeds}")
    print("=" * 96)
    hdr = f"{'池 N':>7} {'候选集':>9} {'胜者':>4} | " + " | ".join(
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
            # 直接由逐用户名次复算，不依赖 experiment.json 的舍入值
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

    # 倍数表（跨档位可比的口径）
    print()
    print(f"{'池 N':>7} | " + " | ".join(f"{e:>13}" for e in ENGINES))
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
    print("新颖度分布（长尾判定的依据）")
    print(f"  公式校验最大偏差 = {nv['formula_max_dev']:.1e}  → 通过")
    print(f"  作品 {nv['n_items']:,} 部   新颖度 均值 {nv['novel_mean']:.4f} "
          f"中位数 {nv['novel_median']:.4f}   长尾阈值 = 中位数 = {nv['novel_median']:.4f}")
    print(f"  长尾作品 {nv['n_longtail']:,} 部（{nv['longtail_share']*100:.1f}%）"
          f"   零人气 {nv['n_zero_pop']} 部   最热门 {nv['n_hottest']} 部")
    print(f"  头部集中度：前 1% 作品占 {nv['head1_share']*100:.1f}% 人气，"
          f"前 10% 占 {nv['head10_share']*100:.1f}%")

    out = dict(
        source="results/ranks.npz + results/experiment.json + data/ga/item_features.jsonl",
        part="第一部分 · ICF（无重排召回对比）",
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
