# -*- coding: utf-8 -*-
"""
list_metrics.py — 列表级指标与特征预处理（GA 适应度的计算基础）。

GA 适应度 = 0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度，
其中后三项都作用在"整份列表"上，所以需要这里的近似度/聚合函数。

四个量的定义
------------
多样性 diversity  : 1 − 列表内两两 Jaccard 相似度的均值（0.3 权重）
新颖度 novelty    : 列表内条目新颖度 novelty_train 的均值（0.2 权重）
新鲜度 fresh      : 列表内"近 5 年新番"占比（0.1 权重）
相关性 relevance  : 列表内召回分数的归一化均值（0.4 权重，由召回引擎提供）

Jaccard 相似度把每部作品的 (题材 ∪ 主题) 当作集合，
所以"同题材扎堆"的列表多样性低——这正是 GA 要压下去的东西。
"""
import numpy as np


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pool_similarity_matrix(pool_idx, tag_sets):
    """候选池内部两两 Jaccard 相似度矩阵 (n, n)。"""
    n = len(pool_idx)
    M = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            v = jaccard(tag_sets[pool_idx[a]], tag_sets[pool_idx[b]])
            M[a, b] = M[b, a] = v
    return M


def list_metrics(items, tag_sets, novelty, fresh):
    """一份有序列表的 多样性 / 新颖度 / 新鲜度。"""
    if len(items) == 0:
        return dict(diversity=0.0, novelty=0.0, fresh=0.0)
    L = len(items)
    sims = [jaccard(tag_sets[items[a]], tag_sets[items[b]])
            for a in range(L) for b in range(a + 1, L)]
    div = 1.0 - float(np.mean(sims)) if sims else 0.0
    return dict(diversity=div,
                novelty=float(np.mean([novelty[i] for i in items])),
                fresh=float(np.mean([fresh[i] for i in items])))


# ------------------------------------------------------------------ 特征预处理
def build_tag_sets(feats):
    """每部作品的 (题材 ∪ 主题) 去重排序元组，供 Jaccard 使用。"""
    return [tuple(sorted(set(feats["genres"][i]) | set(feats["themes"][i])))
            for i in range(len(feats["genres"]))]


def fresh_flag(years, span=5):
    """'新鲜' = 首播年份落在目录最新 `span` 年内。返回 (0/1 向量, 年份阈值)。"""
    ys = years[years > 0]
    if len(ys) == 0:
        return np.zeros(len(years), dtype=np.float32), 0
    thr = int(ys.max()) - (span - 1)
    return (years >= thr).astype(np.float32), thr
