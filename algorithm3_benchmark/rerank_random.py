# -*- coding: utf-8 -*-
"""
rerank_random.py — 重排层的随机对照（两个口径）。

为什么必须有随机对照
--------------------
只拿"贪心 vs GA"两列，读者无法判断 GA 的收益是**重排能力**换来的，
还是**仅仅因为引入了随机性**（毕竟 GA 里本来就有随机初始化/变异）。
所以必须补上随机臂，而且两个口径的含义完全不同，都要给：

  · 窗口内随机（win）—— 从同一重排窗口（相关性前 20）里随机挑 10 条。
    候选来源与 GA **完全相同**，只把"怎么挑"换成掷骰子。
    → 直接量化 GA 的重排能力：GA 若不能显著超过它，GA 就没有存在价值。
  · 整池随机（pool）—— 完全不看相关性，从整个候选池里随机挑 10 条。
    命中率会塌到 K/N（瞎猜水平），但曝光结构接近全库分布。
    → 给出"曝光结构"的理论上界，说明 GA 是在用命中率换曝光、换了多少。

为什么只有一个函数
------------------
两个口径**算法完全相同**，区别只在于调用方传入哪个池子：

    rerank_random(p_i, p_s, rng)              # 窗口内随机
    rerank_random(CAND[r, :cc[r]], s_r, rng)  # 整池随机

所以这里只保留一个 `rerank_random`，两个臂在 run_experiment.py 里区分。
把"窗口/整池"做成函数参数只会制造一个永远只取两个值的开关，反而更难读。

输出约定（与 rerank_ga 保持一致，否则 HR@5 不可比）
--------------------------------------------------
1. **先随机挑出 k 条，再按相关性降序展示**（`argsort(-score, stable)`）。
   rerank_ga 末尾也做同样的重排，所以三个重排臂的"展示顺序语义"一致，
   HR@5 / nov5 这类依赖顺序的指标才有可比性。
2. 池子不足 k 条时没有自由度，直接返回整个池子（按相关性降序）——
   与 rerank_ga 在 `n_pool <= L` 分支的行为一致。
3. 随机性只来自调用方传入的 `rng`，因此逐用户可精确复算
   （随机流由 candidate_pool.arm_rng 提供）。
"""
import numpy as np

import config as C


def rerank_random(pool_idx, pool_score, rng, k=C.TOP_K_OUT):
    """随机重排。返回 (有序条目, 池内位置)。

    参数
    ----
    pool_idx   : 候选条目索引（**整池**或**重排窗口**，由调用方决定）
    pool_score : 与 pool_idx 对齐的召回分数
    rng        : numpy Generator —— 逐用户确定性随机流
    k          : 挑多少条，默认 TOP_K_OUT = 10
    """
    pool_idx = np.asarray(pool_idx)
    pool_score = np.asarray(pool_score)
    n = pool_idx.size
    rel_order = np.argsort(-pool_score, kind="stable")
    if n <= k:
        # 池子不超过 k 条：无论怎么随机都是整个池子，按相关性降序返回
        return pool_idx[rel_order], rel_order
    pick = rng.choice(n, size=k, replace=False)
    pick = pick[np.argsort(-pool_score[pick], kind="stable")]
    return pool_idx[pick], pick
