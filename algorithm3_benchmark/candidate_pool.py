# -*- coding: utf-8 -*-
"""
candidate_pool.py — 候选池构造 + GA 的逐用户随机流。

这两件事放在一起，是因为它们共同定义了"每个用户在每档池子上看到什么、
GA 用哪条随机流"，也就是实验可复现性的全部来源。

一、候选池口径（推荐系统标准的 sampled-negative）
--------------------------------------------------
每位用户的候选集 = 「藏起的那 1 条正例」+「(N−1) 个从未评过的条目」。
10 档：N = 20 / 50 / 100 / 200 / 500 / 1000 / 2000 / 5000 / 10000 / 全目录。
N 越大干扰项越多、任务越难；全目录档（平均 14,097.9 条）最接近真实部署。

二、抽样算法（确定性、可逐位复现）
----------------------------------
`build_negative_pool` 的顺序就是该用户"未评条目"的一个均匀随机排列：
先按全局排列 perm 洗牌，再剔除已评条目与藏起条目。因为 perm 均匀随机，
洗牌后剩下的序列对该用户仍然均匀随机（与从补集里直接采样等价，
但不需要为每个用户单独构造补集）。

三、GA 随机流（逐用户确定性）
------------------------------
    rng = default_rng([sampling_seed, pool_size, engine_id, user_id])

`grid_key` 必须用**池规模本身**（全目录用哨兵值 FULL_GRID_KEY），不能用网格下标——
否则把网格拆到多个进程并行跑时，同一档池子的随机流会随分组方式改变。
这样"每个用户在每档池子上的 GA"有独立且可重算的随机流，
不受 chunk 切分、队列规模、进程拆分、引擎循环顺序影响。

四、随机对照臂的随机流
----------------------
    rng = default_rng([sampling_seed, pool_size, engine_id, user_id, arm])

比 GA 的流多一个 `arm` 分量（1 = 窗口内随机，2 = 整池随机），
两条流互不相干。**不能**直接在 user_rng 的种子列表后面追加 `arm`——
那会改变 SeedSequence 的熵、让全部历史 GA 结果漂移，报告里的数字就会对不上。
"""
import numpy as np

import config as C

# ------------------------------------------------------------------ 候选池档位
GRID = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, None]
GRID_LABEL = [("full" if g is None else str(g)) for g in GRID]

SAMPLING_SEED_BASE = C.RANDOM_SEED + 7000    # 7042
SCORE_CHUNK = 1000                           # 每批处理多少用户（显存/内存控制）

# ------------------------------------------------------------------ 重排窗口
RERANK_DEPTH = C.RECALL_POOL                 # 20：GA 只看候选打分排序后的前 20 名
FULL_GRID_KEY = 10 ** 7                      # "全目录"这一档的哨兵键

ENGINE_ID = {"ICF": 0, "SVD": 1, "NCF": 2}   # 写死：改顺序会漂移 GA 随机流

# 随机对照臂的编号（arm_rng 的第 5 个种子分量）
ARM_ID = {"rand_window": 1, "rand_pool": 2}


def build_negative_pool(perm, rated_items, hidden):
    """返回该用户「未被排除条目」的一个均匀随机排列（前 N−1 个即负例）。

    perm         : 该种子的全局随机排列（长度 n_items）
    rated_items  : 该用户训练集里评过的 item 索引
    hidden       : 该用户被藏起的那 1 条 item 索引
    """
    excl = np.zeros(perm.size, dtype=bool)
    excl[rated_items] = True
    excl[hidden] = True
    return perm[~excl[perm]]


def user_rng(sampling_seed, grid_key, engine, user_id):
    """逐用户确定性随机流：与 chunk 切分、队列规模、网格拆到哪个进程均无关。"""
    return np.random.default_rng([int(sampling_seed), int(grid_key),
                                  int(ENGINE_ID[engine]), int(user_id)])


def arm_rng(sampling_seed, grid_key, engine, user_id, arm):
    """随机对照臂的逐用户随机流（arm 取 ARM_ID 里的值）。

    ⚠️ 这里**故意不修改 user_rng**、而是多加一个种子分量、另起一条流。
    往 user_rng 的种子列表里追加元素会改变 SeedSequence 的熵，
    从而让**历史上所有 GA 结果整体漂移**——那样报告里的已发布数字就全部对不上了。
    所以对照臂必须用独立的 `arm_rng`。
    """
    return np.random.default_rng([int(sampling_seed), int(grid_key),
                                  int(ENGINE_ID[engine]), int(user_id), int(arm)])


def pool_permutation(seed, n_items):
    """该采样种子下的全局物品排列。"""
    return np.random.default_rng(seed).permutation(n_items)
