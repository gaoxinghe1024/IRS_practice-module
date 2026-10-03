# -*- coding: utf-8 -*-
"""
rerank_greedy.py — 重排层①：贪心基线（纯相关性 Top-K）。

这是 GA 的对照组，也是理解整份报告的关键：
GA 的适应度里 0.4 权重是相关性，而贪心解**只**优化相关性。
所以"贪心 = GA 的起点"——GA 是在贪心解附近，用剩下 0.6 的自由度
（多样性 0.3 + 新颖度 0.2 + 新鲜度 0.1）去换曝光结构。

一个必须写清楚的恒等式
----------------------
召回窗口本身就是按相关性降序排好的（`run_experiment.py` 里有 assert 保证），
而贪心的定义是"纯按相关性取 Top-K"，所以：

    贪心（greedy） ≡ 无重排（no rerank）

两条口径选出的条目集合与顺序完全相同。这不是巧合，而是幻灯片表里
两行都标 baseline 的原因。因此本模块只有一个函数——
"无重排"不需要单独实现，它就是本函数在 K = 窗口深度时的特例。

本报告中"ICF 基线"指的就是这里的 Top-10。
"""
import numpy as np

import config as C


def rerank_greedy(pool_idx, pool_score, k=C.TOP_K_OUT):
    """纯相关性 Top-K。返回 (有序条目, 池内位置)。

    参数
    ----
    pool_idx   : 候选池的条目索引（已按相关性降序排列）
    pool_score : 与 pool_idx 对齐的召回分数
    k          : 取前 k 条，默认 TOP_K_OUT = 10

    因为 pool_score 是降序的，`argsort(-score, kind="stable")` 恒为 [0..n-1]，
    所以结果等于直接取前 k 条；这里仍显式排序，是为了让"纯按相关性排序"
    这一语义在代码里可见，而不是依赖调用方的隐式约定。
    """
    pos = np.argsort(-np.asarray(pool_score), kind="stable")[:k]
    return pool_idx[pos], pos
