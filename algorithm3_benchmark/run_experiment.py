# -*- coding: utf-8 -*-
"""
run_experiment.py — 主实验：跑出报告所需的全部对比数据。

它做什么
--------
在 10 档候选池 × 2 个采样种子 × 20,000 用户上，对 3 个召回引擎做两阶段评测：

  phase A（便宜）  no-rerank 名次：每个引擎在候选池里排到第几名
                   → HR@5 / HR@10，报告第一部分的全部对比数据来源
  phase B（较贵）  对每档的胜者引擎跑**四个重排臂**，得到每位用户的 Top-10
                   → HR@5 / HR@10 ＋ 列表级指标（多样性/新颖度/新鲜度/长尾占比）
                   → 报告第二部分"GA 到底做了什么"的全部数据来源

四个重排臂（同一个候选池、同一条召回结果，只换"怎么挑这 10 条"）
------------------------------------------------------------------
    贪心（greedy）  纯按相关性取 Top-10        ← 也 = 无重排
    窗口随机（randw） 在相关性前 20 名里随机挑 10 条
    整池随机（randp） 完全不看相关性，从整个候选池随机挑 10 条
    GA               相关性 + 多样性 + 新颖度 + 新鲜度 四项加权优化
窗口随机用来证明"GA 的收益来自重排能力而非随机性"；
整池随机给出曝光结构的理论上界（命中率同时塌到 K/N）。

产出
----
    results/ranks.npz       逐用户原始数据（紧凑二进制，报告的唯一数据源）
    results/experiment.json 逐档汇总表 + 三引擎无重排 HR + 协议快照

用法
----
    python run_experiment.py                  # 完整跑（20,000 用户，约 80~90 分钟）
    python run_experiment.py --prefix 300     # 冒烟：只评真实队列前 300 人，约 4 分钟

可复现性
--------
* ICF 是确定性引擎：重跑名次必须与历史结果**逐位一致**，否则脚本直接报错退出。
* GA 的随机流 = default_rng([种子, 池规模, 引擎, 用户])，逐用户独立可复算。
* 随机对照臂用**独立**的 arm_rng（种子末尾多一个 arm 分量），
  保证新增对照臂不会扰动任何已发布的 GA 数字。
* `--prefix` 取的是**真实 20,000 人队列的前 N 位**（而非另抽一个小队列），
  因此冒烟结果可以与全量结果逐用户对照。
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

# 需要落盘的指标：GA 侧与两个随机臂多两项命中率
GA_METRICS = ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")
GREEDY_METRICS = ("nov5", "nov10", "div10", "fresh10", "lt10")
RANDOM_METRICS = ("hr5", "hr10", "nov5", "nov10", "div10", "fresh10", "lt10")



def log(msg):
    print(msg, flush=True)


def _arm_metrics(items, tag_sets, novelty, fresh, nov_thr, hidden):
    """一份 Top-10 列表的 7 项指标 —— 四个重排臂共用同一套度量代码。

    共用是刻意的：只有四臂走**逐字相同**的度量实现，
    "贪心 / 窗口随机 / 整池随机 / GA"之间的差异才能唯一归因到重排策略本身。
    """
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
                    help=">0 时只评真实队列的前 N 位用户（冒烟/位级校验用）")
    ap.add_argument("--tag", default="",
                    help="给输出加后缀（如 --tag smoke 会写出 ranks_smoke.npz），"
                         "避免冒烟运行覆盖正式结果")
    args = ap.parse_args()

    t_all = time.time()
    grid, grid_label = CP.GRID, CP.GRID_LABEL
    n_grid = len(grid)

    log("=" * 78)
    log("ICF + GA 主实验：10 档候选池 × 3 召回引擎 × 4 重排臂")
    log("=" * 78)

    # ------------------------------------------------------------ 1. 数据
    log("\n[1/5] 加载数据 ...")
    R = D.load_ratings_train()
    a2i, u2i = D.load_mappings()
    n_users, n_items = R.shape
    val = D.load_validation()
    v = D.build_validation_indexed(val, a2i, u2i)
    hidden_all = D.build_eligible_cohort(R, v)
    cohort = D.sample_cohort(hidden_all, size=args.users)
    if args.prefix:
        cohort = cohort.iloc[:args.prefix].reset_index(drop=True)
        log(f"  --prefix {args.prefix}：只评真实队列前 {args.prefix} 位用户")
    cohort_u = cohort["u"].to_numpy().astype(np.int64)
    hidden_i = cohort["i"].to_numpy().astype(np.int64)
    n = len(cohort_u)
    log(f"  队列={n:,}  作品={n_items:,}  训练评分={R.nnz:,}")

    feats = D.load_item_features(a2i, n_items)
    years = D.load_release_years(a2i, n_items)
    tag_sets = LM.build_tag_sets(feats)
    novelty = feats["novelty"]
    fresh, fresh_thr = LM.fresh_flag(years, span=5)
    seen = feats["seen"]
    nov_thr = float(np.median(novelty[seen]))
    log(f"  长尾判定阈值 = 有元数据作品的新颖度中位数 = {nov_thr:.4f}")

    perms = {s: CP.pool_permutation(s, n_items) for s in SEEDS}

    # ------------------------------------------------------------ 2. 引擎
    log("\n[2/5] 训练引擎（ICF 确定性；SVD/NCF 固定种子重训）...")
    engines = REG.load_engines(ENGINES, R, n_items)

    ns = len(SEEDS)
    rank_nr = {e: np.zeros((ns, n_grid, n), np.int32) for e in ENGINES}
    cand_count = np.zeros((ns, n_grid, n), np.int32)
    ga_m = {e: {m: np.zeros((ns, n_grid, n), np.float32) for m in GA_METRICS}
            for e in ENGINES}
    gr_m = {e: {m: np.zeros((ns, n_grid, n), np.float32) for m in GREEDY_METRICS}
            for e in ENGINES}
    # 随机对照臂：只对每档的胜者引擎落盘（与 GA/贪心 的评估对象完全一致），
    # 所以不再按引擎分桶，避免 2/3 的存储是零。每档用的是哪个引擎记在 winner_engine。
    randw_m = {m: np.zeros((ns, n_grid, n), np.float32) for m in RANDOM_METRICS}
    randp_m = {m: np.zeros((ns, n_grid, n), np.float32) for m in RANDOM_METRICS}

    # ------------------------------------------------------ phase A：无重排名次
    log("\n[3/5] phase A：no-rerank 名次（HR@5 / HR@10 的来源）...")
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
        log(f"  chunk {s0:>6}-{e0:<6} 完成（累计 {time.time()-t0:.0f}s）")

    # ---- 位级自检：若 results/ranks.npz 已存在且形状一致，ICF 必须逐位相同
    stored = os.path.join(C.OUT_DIR, "ranks.npz")
    if os.path.exists(stored):
        zs = np.load(stored, allow_pickle=True)
        for e in ENGINES:
            key = f"rank_nr_{e}"
            if key not in zs.files:
                continue
            sr = zs[key]
            if sr.shape != rank_nr[e].shape:
                log(f"  [{e}] 已存形状 {sr.shape} != 本次 {rank_nr[e].shape}（队列规模不同，跳过位级自检）")
                continue
            same = np.array_equal(sr, rank_nr[e])
            log(f"  [{e}] 与已存名次逐位一致: {same}")
            if e == "ICF" and not same:
                raise SystemExit("!! ICF 必须逐位一致，出现偏差说明复现环境变了，停止。")

    # ---- 每档胜者（无重排 HR@5 口径；SVD 从不胜出）
    winner = {}
    for gi, gl in enumerate(grid_label):
        vals = {e: float((rank_nr[e][:, gi, :] < C.HR_K).mean()) for e in ENGINES}
        winner[gi] = max(("ICF", "NCF"), key=lambda e: vals[e])
        log(f"  池 {gl:>5}: 胜者 {winner[gi]}   "
            + "  ".join(f"{e}={vals[e]:.5f}" for e in ENGINES))

    # ------------------------------------- phase B：GA 重排 + 列表级指标
    log("\n[4/5] phase B：对胜者引擎跑四个重排臂（贪心/窗口随机/整池随机/GA）...")
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
                gk = CP.FULL_GRID_KEY if g is None else int(g)   # 随机流用池规模，非网格下标
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
                # 贪心臂的 HR 不单独算：它恒等于 phase A 的无重排名次（见 rerank_greedy.py）
                rm = {k: np.zeros(m, np.float32) for k in GREEDY_METRICS}
                wm = {k: np.zeros(m, np.float32) for k in RANDOM_METRICS}
                pm = {k: np.zeros(m, np.float32) for k in RANDOM_METRICS}
                for r in range(m):
                    h = int(hi[r])
                    d = int(depth_arr[r])
                    uid = int(uu[r])
                    s_r = cand_sc[r, :cc[r]]
                    # 分数降序窗口（与主实验完全一致的取法）
                    if d <= C.HR_K:
                        sel = np.argsort(-s_r, kind="stable")[:d]
                    else:
                        sel = np.argpartition(-s_r, d - 1)[:d]
                        sel = sel[np.argsort(-s_r[sel], kind="stable")]
                    p_i = CAND[r, sel]
                    p_s = s_r[sel]
                    assert np.all(np.diff(p_s.astype(np.float64)) <= 0), \
                        "重排窗口必须已按相关性降序"
                    # ---------------- 四个重排臂：同一个池子，只换"怎么挑这 10 条"
                    # ① GA
                    if d <= C.HR_K:
                        # 窗口深度不足时 GA 没有自由度：GA 列表 = 窗口本身
                        ga_items = list(p_i)
                    else:
                        items, _, _ = rerank_ga(
                            p_i, p_s, tag_sets, novelty, fresh,
                            np.zeros(len(p_i), dtype=np.float32),
                            CP.user_rng(s, gk, w, uid))
                        ga_items = list(items)
                    # ② 贪心（纯相关性 Top-10，= 无重排 Top-10）
                    greedy_items = list(rerank_greedy(p_i, p_s)[0])
                    # ③ 窗口随机：在相关性前 20 名里随机挑 10 条
                    win_items = list(rerank_random(
                        p_i, p_s,
                        CP.arm_rng(s, gk, w, uid, CP.ARM_ID["rand_window"]))[0])
                    # ④ 整池随机：完全不看相关性，从候选池随机挑 10 条
                    pool_items = list(rerank_random(
                        CAND[r, :cc[r]], s_r,
                        CP.arm_rng(s, gk, w, uid, CP.ARM_ID["rand_pool"]))[0])
                    # 同一套度量代码，四臂各落一份
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
        log(f"  chunk {s0:>6}-{e0:<6} 完成（累计 {time.time()-t1:.0f}s）")

    # ------------------------------------------------------------ 汇总
    log("\n[5/5] 汇总 ...")
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
        # 随机对照臂（窗口内 / 整池）
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
            # 自检：整池随机臂实测命中率应当 == 解析值 K/N（误差仅来自 float 精度）
            randp_hr10_gap=round(abs(rp10 - rand10), 8),
        )
        row["d_nov10"] = round(row["ga_nov10"] - row["greedy_nov10"], 4)
        row["d_nov5"] = round(row["ga_nov5"] - row["greedy_nov5"], 4)
        row["d_lt10"] = round(row["ga_lt10"] - row["greedy_lt10"], 4)
        row["d_div10"] = round(row["ga_div10"] - row["greedy_div10"], 4)
        # 「GA 的重排能力」= GA 相对**窗口随机**的净增益（扣掉"随机性本身"的贡献）
        row["ga_over_randw_nov10"] = round(row["ga_nov10"] - row["randw_nov10"], 4)
        row["ga_over_randw_lt10"] = round(row["ga_lt10"] - row["randw_lt10"], 4)
        row["ga_over_randw_hr10"] = round(row["ga_hr10"] - row["randw_hr10"], 5)
        tables[gl] = row
        log(f"  {gl:>5} [{w}]  HR@10  贪心={nr10:.4f} 窗随={rw10:.4f} 池随={rp10:.4f} GA={ga10:.4f}"
            f"   长尾占比  贪心={row['greedy_lt10']:.4f} 窗随={row['randw_lt10']:.4f} "
            f"池随={row['randp_lt10']:.4f} GA={row['ga_lt10']:.4f}")
        log(f"          整池随机自检 |实测 HR@10 − 解析 K/N| = {row['randp_hr10_gap']:.2e}")

    engines_nr = {}
    for e in ENGINES:
        engines_nr[e] = dict(
            hr5=[round(float((rank_nr[e][:, gi, :] < C.HR_K).mean()), 5) for gi in range(n_grid)],
            hr10=[round(float((rank_nr[e][:, gi, :] < 10).mean()), 5) for gi in range(n_grid)],
        )

    out = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        purpose="ICF + GA 主实验：10 档候选池上的召回对比与 GA 重排收益",
        protocol=dict(
            users=int(n), seeds=SEEDS,
            sampling="prefix (identical to main run)",
            rerank_depth=CP.RERANK_DEPTH, top_k_out=C.TOP_K_OUT,
            rerank_arms=["greedy 贪心＝纯相关性 Top-10（≡ 无重排）",
                         "randw 窗口随机＝在相关性前 20 名内随机挑 10",
                         "randp 整池随机＝完全不看相关性，从整个候选池随机挑 10",
                         "ga 遗传算法＝四项加权适应度优化"],
            ga_rng="default_rng([sampling_seed, pool_size, engine_id, user_id])",
            random_arm_rng=("default_rng([sampling_seed, pool_size, engine_id, user_id, arm])"
                            "，arm=1 窗口随机 / arm=2 整池随机；独立于 GA 流，"
                            "故新增对照臂不会扰动任何已发布的 GA 数字"),
            arm_display_order="所有臂都「先挑出 10 条，再按相关性降序展示」，与 GA 输出约定一致",
            ga_fitness="0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度（幻灯片口径）",
            ga_weights=dict(relevance=C.GA_W_RELEVANCE, diversity=C.GA_W_DIVERSITY,
                            novelty=C.GA_W_NOVELTY, fresh=C.GA_W_FRESH),
            ga_rel_norm="相关性项在池内 min-max 归一化到 [0,1] 后加权，四项才可比",
            novelty_item="novelty_train = 1 - log(1+人气)/log(1+最大人气)（越大越冷门）",
            longtail_threshold=round(nov_thr, 4),
            longtail_def="前 10 条中 novelty >= 全库中位数的条目占比",
            randp_check="整池随机臂实测 HR 应与解析值 K/N 相等（见 tables[*].randp_hr10_gap）",
            engines_retrained=["ICF（确定性，逐位校验）", "SVD（固定种子重训）", "NCF（固定种子重训）"],
            winner_rule="每档无重排 HR@5 在 {ICF, NCF} 中取较大者（SVD 从不在任何池上胜出）"),
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
    log(f"\n总耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
