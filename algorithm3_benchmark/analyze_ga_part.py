# -*- coding: utf-8 -*-
"""
analyze_ga_part.py — 生成报告"第二部分 · 重排对照"的全部对比数据。

回答的问题
----------
"加上 GA 重排之后，推荐列表到底变了什么？好处多大、代价多少？"

四个重排臂（同一个候选池、同一条召回结果，只换"怎么挑这 10 条"）
------------------------------------------------------------------
    greedy  贪心      纯按相关性取 Top-10（≡ 无重排）
    randw   窗口随机  在相关性前 20 名里随机挑 10 条
    randp   整池随机  完全不看相关性，从整个候选池随机挑 10 条
    ga      GA        0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度

为什么随机臂必须有**两个**
--------------------------
* 窗口随机与 GA 共享同一个候选窗口，**只把"怎么挑"换成掷骰子**。
  所以 GA 相对窗口随机的增益 = GA 的重排能力（扣掉了"随机性本身"的贡献）。
  只跟贪心比是不够的：GA 里有随机初始化与变异，必须证明它的收益不是白捡的随机性。
* 整池随机完全不看相关性，命中率会塌到 K/N（瞎猜水平），但曝光结构接近全库分布。
  它给出"曝光结构"的理论上界，用来回答"GA 是在用多少命中率换多少曝光"。

为什么不能只看 nov@10
---------------------
原来的收益指标是 nov@10（前 10 条的新颖度均值），它有三个毛病：
  ① 是 0~1 的小数，"+0.008"读不出体感；
  ② 是**均值**，掩盖了分布——只有少数用户真正被推到了长尾；
  ③ 把"往长尾挪一点点"和"真的推出长尾商品"混为一谈。
所以本脚本把收益翻译成三个体感明确的指标：

  GOS@10  GA 目标分  = 0.3×多样性@10 + 0.2×新颖度@10 + 0.1×新鲜度@10
                       正好是 GA 适应度里**由 GA 负责优化的那 60%**
                       （剩下 0.4 是相关性，由召回引擎保证、用 HR 监控代价）
  LTE@10  长尾曝光量  = Σ_users(lt10 × 10)，全站 20 万个推荐位中长尾商品占的**条数**
  LTR@10  长尾触达率  = 前 10 条里至少出现 1 个长尾商品的用户占比

代价侧仍用 HR@10 / HR@5（对应适应度里 0.4 权重的相关性项）。

配对协议
--------
**同用户、同候选池、同引擎、同随机种子**逐用户做差。
收益指标（连续）用 Wilcoxon 符号秩检验，命中率（二值）用 McNemar 检验，
均给出 95% 置信区间与 Cohen's d_z。

数据来源：results/ranks.npz
输出：results/ga_part.json
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
GENERATIVE = ("randw", "randp")          # 相对贪心"生成式"改变列表结构的两臂

W_REL = C.GA_W_RELEVANCE
W_DIV = C.GA_W_DIVERSITY
W_NOV = C.GA_W_NOVELTY
W_FRESH = C.GA_W_FRESH

TOP_K = 10                 # 每用户展示 10 条 → 10 个曝光位
LONGTAIL_MEDIAN = 0.6156   # 全库新颖度中位数（长尾判定阈值，来自数据包 item_features）

ARM_DEF = {
    "greedy": "贪心：纯按相关性取 Top-10（恒等于无重排）",
    "randw": "窗口随机：在相关性前 20 名里随机挑 10 条",
    "randp": "整池随机：完全不看相关性，从整个候选池随机挑 10 条",
    "ga": "GA：0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度",
}


# ------------------------------------------------------------------ 统计工具
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
    """a=新臂命中, b=对照臂命中（逐用户 0/1）。返回 (p, 对照错&新对, 对照对&新错)。"""
    a = a > 0.5
    b = b > 0.5
    n01 = int((~b & a).sum())
    n10 = int((b & ~a).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    z = (abs(n01 - n10) - 1.0) / sqrt(n)
    return float(erfc(abs(z) / sqrt(2.0))), n01, n10


# ------------------------------------------------------------ 四个臂的统一取数
def mean_series(z, gi, w, arm, key):
    """该臂在某档上的「跨种子平均」逐用户序列。

    取数顺序刻意与历史版本逐字一致（非 HR 键先 mean 再 astype，HR 键先 astype 再 mean），
    这样 greedy / ga 两列能逐位复现已发布报告里的数字。
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
    """逐种子的是否"至少获得 1 条长尾曝光"，形状 (n_seeds, N)。

    必须先逐种子判 >0 再跨种子平均，不能反过来——否则会把"只在其中一个种子
    被曝光"的用户也算成已触达，系统性高估触达率。
    """
    key = {"greedy": f"greedy_{w}_lt10", "ga": f"ga_{w}_lt10",
           "randw": "randw_lt10", "randp": "randp_lt10"}[arm]
    return z[key][:, gi, :] > 0


def user_novelty_mean(z):
    """用户真实消费作品的平均新颖度 —— 用被藏起的那 1 条（真实高分交互）代表。

    这个数必须从数据算、不能凭印象写：它是"GA 的收益不可能无限大"这句结论的
    量化依据（用户自己也在消费头部作品，推荐长尾等于偏离用户历史偏好）。
    """
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
    """一个重排臂在某档上的收益 / 代价快照。"""
    S = {k: mean_series(z, gi, w, arm, k)
         for k in ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")}
    div, nov, fre, lt = S["div10"], S["nov10"], S["fresh10"], S["lt10"]
    ltseed = ltr_seed(z, gi, w, arm)

    gos = W_DIV * div + W_NOV * nov + W_FRESH * fre       # GA 目标分（其负责的 60%）
    lte = float(lt.sum() * TOP_K)                         # 长尾曝光量（条）
    ltr = float(ltseed.mean(0).mean())                    # 长尾触达率

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
    """新臂 a 相对对照臂 b 的收益/代价对照（逐用户配对）。"""
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
    print("第二部分 · 重排对照：贪心 / 窗口随机 / 整池随机 / GA")
    print(f"种子 {seeds}   用户 {N:,}   候选池档位 {grid}")
    print("=" * 118)

    # 每档胜者引擎：无重排 HR@5 在 {ICF, NCF} 中取较大者（与主实验口径一致）
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
    print(f"[新颖度] 长尾阈值（全库中位数）={nvu['threshold']:.4f}   "
          f"用户真实消费作品的平均新颖度={nvu['mean']:.4f}（中位数 {nvu['median']:.4f}，"
          f"{nvu['longtail_share']*100:.1f}% 属长尾）")
    print()

    # 自检①：贪心 HR@10 ≡ 无重排名次 < 10
    gi_full = grid.index("full")
    chk = float((z["rank_nr_ICF"][:, gi_full, :] < 10).mean())
    print(f"[自检①] 全目录档 ICF 贪心 HR@10（由名次复算）= {chk:.5f}"
          f"  → {'通过' if abs(chk - 0.20925) < 1e-4 else '与历史值 0.20925 不一致!'}")
    # 自检②：整池随机臂的实测命中率应等于解析值 K/N（"瞎猜"水平）
    ccf = z["cand_count"][:, gi_full, :].astype(np.float64)
    ana10 = float((np.minimum(10, ccf) / np.maximum(ccf, 1.0)).mean())
    mea10 = float((z["randp_hr10"][:, gi_full, :] > 0.5).astype(np.float64).mean())
    print(f"[自检②] 全目录档 整池随机 HR@10 实测 {mea10:.6f} vs 解析 K/N {ana10:.6f}"
          f"  → {'通过' if abs(mea10 - ana10) < 2e-4 else '不一致!'}")
    print()

    hdr = (f"{'池 N':>7} {'引擎':>4} | {'HR@10':>6} {'贪心':>7} {'窗随':>7} {'池随':>7} {'GA':>7} | "
           f"{'长尾占比 lt10':>13} {'贪心':>7} {'窗随':>7} {'池随':>7} {'GA':>7} | "
           f"{'LTE@10 贪心':>11} {'GA':>8}")
    print(hdr)
    print("-" * 118)

    rows = {}
    for gi, gl in enumerate(grid):
        w = winner[gi]
        A = {a: arm_block(z, gi, w, a, N) for a in ARMS}
        g, rw, rp, gaa = A["greedy"], A["randw"], A["randp"], A["ga"]
        rec = dict(pool=gl, engine=w,
                   n_candidates=float(z["cand_count"][:, gi, :].mean()))

        # ---------- 整池随机自检：实测 HR@10 必须等于解析值 min(K,N)/N
        ccf = z["cand_count"][:, gi, :].astype(np.float64)
        rec["randp_check"] = dict(
            analytic=float((np.minimum(TOP_K, ccf) / np.maximum(ccf, 1.0)).mean()),
            measured=float((z["randp_hr10"][:, gi, :] > 0.5).astype(np.float64).mean()),
        )

        # ---------- 四臂快照（报告主表直接读这里）
        rec["arms"] = {a: dict(
            label=A[a]["label"], lte=A[a]["lte"], ltr=A[a]["ltr"],
            ltr_users=A[a]["ltr"] * N, lte_per_user=A[a]["lte_per_user"],
            gos=A[a]["gos_mean"], hr5=A[a]["hr5_mean"], hr10=A[a]["hr10_mean"],
            nov5=A[a]["nov5_mean"], nov10=A[a]["nov10_mean"],
            div10=A[a]["div10_mean"], fresh10=A[a]["fresh10_mean"],
            lt10=A[a]["lt10_mean"],
        ) for a in ARMS}

        # ---------- GA vs 贪心（历史口径，与已发布报告逐位一致）
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

        # ---------- 代价侧：GA vs 贪心（HR@5 / HR@10）
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

        # ---------- 逐指标配对（GA vs 贪心）
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

        # ---------- 新增配对：GA vs 窗口随机 / GA vs 整池随机 / 随机臂 vs 贪心
        rec["vs_randw"] = pair(gaa, rw, N)
        rec["vs_randp"] = pair(gaa, rp, N)
        rec["randw_vs_greedy"] = pair(rw, g, N)
        rec["randp_vs_greedy"] = pair(rp, g, N)
        # GA 拿到了"整池随机"那部分曝光红利的多少份额（LTE 增益口径）
        gain_randp = rp["lte"] - g["lte"]
        rec["share_of_randp_lte"] = float((gaa["lte"] - g["lte"]) / gain_randp) if gain_randp else float("nan")

        # 收益/代价换算：每损失 1pp HR@10，换来多少 % 的长尾曝光
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
        """基数为 0 时相对涨幅是 0/0，打 nan 会污染日志——直接写 '基数=0'。"""
        return "基数=0" if x != x else f"{x:+.0f}%"

    pos = [g for g in grid
           if rows[g]["benefit"]["lte"]["rel_pct"] == rows[g]["benefit"]["lte"]["rel_pct"]
           and rows[g]["benefit"]["lte"]["rel_pct"] > 0]
    print("\n" + "=" * 118)
    print(f"长尾曝光量 LTE@10 提升的档数（GA vs 贪心）：{len(pos)}/{len(grid)}")
    print("逐档 LTE 提升：" + ", ".join(
        f"{g}:{rel_txt(rows[g]['benefit']['lte']['rel_pct'])}" for g in grid))
    print("逐档 LTR 提升：" + ", ".join(
        f"{g}:{rel_txt(rows[g]['benefit']['ltr']['rel_pct'])}" for g in grid))
    print("逐档 GOS 提升：" + ", ".join(
        f"{g}:{rows[g]['benefit']['gos']['rel_pct']:+.1f}%" for g in grid))
    print("逐档 ΔHR@10：" + ", ".join(
        f"{g}:{rows[g]['tradeoff']['d_hr10_pp']:+.2f}pp" for g in grid))

    full = rows["full"]
    fa = full["arms"]
    print("\n四臂对照（全目录档）")
    for a in ARMS:
        print(f"  {a:<7} HR@10={fa[a]['hr10']*100:7.3f}%  lt10={fa[a]['lt10']:.4f}  "
              f"LTE={fa[a]['lte']:9.0f}  LTR={fa[a]['ltr']*100:6.2f}%  GOS={fa[a]['gos']:.4f}")
    print(f"  GA vs 贪心    ：LTE {full['benefit']['lte']['rel_pct']:+.1f}%  "
          f"ΔHR@10 {full['tradeoff']['d_hr10_pp']:+.2f}pp")
    print(f"  GA vs 窗口随机：LTE {full['vs_randw']['lte']['rel_pct']:+.1f}%  "
          f"ΔHR@10 {full['vs_randw']['hr10']['delta_pp']:+.2f}pp  "
          f"p(McNemar)={full['vs_randw']['hr10']['p']:.3g}")
    print(f"  GA vs 整池随机：LTE {full['vs_randp']['lte']['rel_pct']:+.1f}%  "
          f"ΔHR@10 {full['vs_randp']['hr10']['delta_pp']:+.2f}pp  "
          f"→ GA 拿到整池随机曝光红利的 {full['share_of_randp_lte']*100:.1f}%")

    print("\n整池随机臂自检（实测 HR@10 应等于解析值 min(K,N)/N）")
    worst = 0.0
    for g in grid:
        rc = rows[g]["randp_check"]
        dev = abs(rc["measured"] - rc["analytic"])
        worst = max(worst, dev)
        print(f"  {g:>7} 解析 {rc['analytic']:.6f}   实测 {rc['measured']:.6f}   "
              f"偏差 {dev:.2e}")
    print(f"  → 全部 {len(grid)} 档最大偏差 {worst:.2e}"
          f"  {'通过' if worst < 5e-3 else '偏大，需检查随机臂实现!'}")

    print("\n四臂全目录汇总（报告 4.1 主表用）")
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
        # ---- 四臂对照（报告主表用）
        four_arm_full={a: dict(
            label=fa[a]["label"], hr5=fa[a]["hr5"] * 100, hr10=fa[a]["hr10"] * 100,
            lt10=fa[a]["lt10"], lte=fa[a]["lte"], lte_per_user=fa[a]["lte_per_user"],
            ltr=fa[a]["ltr"] * 100, ltr_users=fa[a]["ltr_users"], gos=fa[a]["gos"],
            nov5=fa[a]["nov5"], nov10=fa[a]["nov10"], div10=fa[a]["div10"],
            fresh10=fa[a]["fresh10"]) for a in ARMS},
        # ---- 关键对照量
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
