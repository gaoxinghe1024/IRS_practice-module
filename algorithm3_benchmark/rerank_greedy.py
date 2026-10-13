# -*- coding: utf-8 -*-
"""
rerank_greedy.py — Rerank layer 1: greedy baseline (pure relevance Top-K).

The recall window is already sorted by descending relevance (asserted in
run_experiment.py), so greedy here is identical to no rerank: both pick the
same items in the same order. This is the GA's internal baseline — the
"ICF baseline" in the report is this Top-10.
"""
import numpy as np

import config as C


def rerank_greedy(pool_idx, pool_score, k=C.TOP_K_OUT):
    """Pure relevance Top-K. Returns (ordered items, positions within the pool).

    pool_idx   : candidate pool item indices (already sorted by descending relevance)
    pool_score : recall scores aligned with pool_idx
    k          : take the top k, default TOP_K_OUT = 10

    pool_score is descending, so argsort(-score, kind="stable") is always
    [0..n-1]; the explicit sort keeps the "sorted purely by relevance"
    semantics visible in code instead of relying on the caller.
    """
    pos = np.argsort(-np.asarray(pool_score), kind="stable")[:k]
    return pool_idx[pos], pos
