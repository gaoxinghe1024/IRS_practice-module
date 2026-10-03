# -*- coding: utf-8 -*-
"""
rerank_ga.py — 重排层②：遗传算法（GA）。

适应度（幻灯片口径）
--------------------
    Fitness = 0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度
"它的内部基线是贪心算法（纯按相关性取 Top-N），GA 在此基础上优化其余三项。"

编码方式：随机键（random keys）
-------------------------------
每个个体是一个长度 = 池规模的实数键向量，选出的 Top-K 子集 = `argsort(-keys)[:K]`。
这样**任何键向量都对应一个合法子集**，不需要"去重修复"步骤，
整个种群可以向量化评估——这是 2 万用户 × 10 档池子跑得下来的关键。

与 SPEC 的冲突说明
------------------
SYSTEM_DEV_HANDOFF.md 记录的适应度是
    0.4 相关性 + 0.3 个性化 + 0.2 多样性 + 0.1 新颖度
与幻灯片不同。本实验以**幻灯片**为准（GA_FITNESS_MODE = "slide"），
"handoff" 口径仅留作敏感性分析。

逐用户确定性
------------
GA 的随机性完全来自调用方传入的 `rng`（见 candidate_pool.user_rng），
所以同一用户在同一档池子上的 GA 结果可精确复算。
"""
import numpy as np

import config as C
from list_metrics import pool_similarity_matrix


def _fitness(pop, rel_norm, novelty, fresh, pers, pool_sim, mode):
    """对整个解码后的种群做向量化适应度评估。

    pop : (N, L) 池内位置下标；其余向量均按池对齐。
    返回（加权总分, 四项分量明细）。
    """
    rel = rel_norm[pop].mean(1)
    nov = novelty[pop].mean(1)
    fr = fresh[pop].mean(1)
    L = pop.shape[1]
    sub = pool_sim[pop[:, :, None], pop[:, None, :]]        # (N, L, L)
    iu = np.triu_indices(L, k=1)
    div = 1.0 - sub[:, iu[0], iu[1]].mean(1)
    pe = pers[pop].mean(1)
    if mode == "handoff":
        # SYSTEM_DEV_HANDOFF.md §5
        total = 0.4 * rel + 0.3 * pe + 0.2 * div + 0.1 * nov
    else:
        # 幻灯片：0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度
        total = (C.GA_W_RELEVANCE * rel + C.GA_W_DIVERSITY * div
                 + C.GA_W_NOVELTY * nov + C.GA_W_FRESH * fr)
    return total, dict(relevance=rel, diversity=div, novelty=nov,
                       fresh=fr, personalization=pe)


def rerank_ga(pool_idx, pool_score, tag_sets, novelty, fresh, pers, rng,
              mode=C.GA_FITNESS_MODE):
    """对候选池跑 GA。返回 (有序条目列表, 池内位置, 最优适应度)。"""
    n_pool = len(pool_idx)
    L = min(C.TOP_K_OUT, n_pool)
    if n_pool <= L:
        pos = np.arange(n_pool)
        return pool_idx[pos], pos, None

    # 相关性在池内归一化到 [0,1]，四个权重才可比
    s = np.asarray(pool_score, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    rel_norm = ((s - lo) / (hi - lo) if hi > lo else np.ones_like(s)).astype(np.float32)

    nov_p = np.asarray([novelty[i] for i in pool_idx], dtype=np.float32)
    fre_p = np.asarray([fresh[i] for i in pool_idx], dtype=np.float32)
    per_p = np.asarray(pers, dtype=np.float32)
    if per_p.shape[0] != n_pool:
        per_p = np.zeros(n_pool, dtype=np.float32)
    sim = pool_similarity_matrix(pool_idx, tag_sets)

    N = C.GA_POP
    keys = rng.random((N, n_pool)).astype(np.float32)
    # 用贪心解播种一个个体：严格降序的键
    keys[0] = np.linspace(1.0, 0.0, n_pool, dtype=np.float32)

    def decode(k):
        return np.argsort(-k, axis=1)[:, :L]

    pop = decode(keys)
    fit, _ = _fitness(pop, rel_norm, nov_p, fre_p, per_p, sim, mode)
    b = int(np.argmax(fit))
    best, best_fit = pop[b].copy(), float(fit[b])

    n_mut = max(1, int(C.GA_MUTATION * N))
    for _ in range(C.GA_GENERATIONS):
        # --- 锦标赛选择
        cand = rng.integers(0, N, size=(N, C.GA_TOURNAMENT))
        winner = cand[np.arange(N), np.argmax(fit[cand], axis=1)]
        new = keys[winner].copy()
        # --- 键向量上的均匀交叉
        mask = rng.random((N, n_pool)) < C.GA_CROSSOVER
        partner = keys[rng.integers(0, N, size=N)]
        new = np.where(mask, new, partner).astype(np.float32)
        # --- 精英保留
        keep = np.argsort(-fit)[:C.GA_ELITE]
        new[:C.GA_ELITE] = keys[keep]
        # --- 变异：扰动一部分个体的一个键
        rows = rng.choice(N, size=n_mut, replace=False)
        new[rows, rng.integers(0, n_pool, size=n_mut)] = rng.random(n_mut).astype(np.float32)
        keys = new

        pop = decode(keys)
        fit, _ = _fitness(pop, rel_norm, nov_p, fre_p, per_p, sim, mode)
        b = int(np.argmax(fit))
        if fit[b] > best_fit:
            best_fit, best = float(fit[b]), pop[b].copy()

    # 输出顺序仍按相关性降序（保证"命中位置越靠前越好"的语义）
    order = best[np.argsort(-rel_norm[best], kind="stable")]
    return pool_idx[order], order, best_fit
