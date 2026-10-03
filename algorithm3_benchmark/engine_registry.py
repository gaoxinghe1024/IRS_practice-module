# -*- coding: utf-8 -*-
"""
engine_registry.py — 三个召回引擎的统一加载与打分接口。

为什么需要这一层
----------------
三个引擎的实现风格不同（ICF 是稀疏矩阵、SVD/NCF 是 torch 模型），
但实验只关心两件事：① 加载并训练；② 给定用户给出整目录打分。
把它们统一到 `load_engines` + `full_scores` 两个函数，
实验脚本就不需要知道任何引擎内部细节。

三个引擎
--------
    ICF  物品协同过滤：物品-物品相似度，确定性引擎（重训逐位一致）
    SVD  矩阵分解：用户/作品共享 32 维隐空间
    NCF  神经协同过滤：MLP + 负采样，隐向量 32 维

注意：ICF 的确定性使它成为整条管线的"校验锚"——
重跑后 ICF 的名次必须与历史结果逐位一致，否则说明环境变了。
"""
import time

import numpy as np

import config as C

ENGINES = ["ICF", "SVD", "NCF"]


def load_engines(names, R, n_items):
    """训练并返回 {引擎名: 引擎对象}。R 是训练评分 CSR。"""
    out = {}
    for name in names:
        t = time.time()
        print(f"\n--- training {name} ---", flush=True)
        if name == "ICF":
            from engine_icf import ICFEngine
            eng = ICFEngine()
        elif name == "SVD":
            from engine_svd import SVDEngine
            eng = SVDEngine()
        elif name == "NCF":
            from engine_ncf import NCFEngine
            eng = NCFEngine()
        else:
            raise ValueError(name)
        eng.fit(R)
        eng.n_items = n_items
        print(f"  {name} fit {time.time()-t:.1f}s", flush=True)
        out[name] = eng
    return out


def full_scores(eng, uu):
    """整目录打分：(n_users, n_items) float32。只用于确定排序，不做 clip。"""
    if hasattr(eng, "score_items"):
        return np.asarray(eng.score_items(uu), dtype=np.float32)
    n = len(uu)
    uu_rep = np.repeat(np.asarray(uu, dtype=np.int64), eng.n_items)
    ii_rep = np.tile(np.arange(eng.n_items, dtype=np.int64), n)
    return eng.predict_pairs(uu_rep, ii_rep, clip=False).reshape(n, eng.n_items)
