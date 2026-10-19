# -*- coding: utf-8 -*-
"""
rerank_random.py — Random control arms of the rerank layer (two protocols).

Window random: pick 10 items at random from the same rerank window (top-20
by relevance) as GA — isolates GA's reranking ability from randomness itself.
Pool random: pick 10 items at random from the whole candidate pool — hit rate
collapses to K/N (blind-guess level) but exposure structure is closest to the
catalogue distribution. Both protocols share this single function; the caller
chooses which pool to pass (see run_experiment.py).
"""
import numpy as np

import config as C


def rerank_random(pool_idx, pool_score, rng, k=C.TOP_K_OUT):
    """Random rerank. Returns (ordered items, positions within the pool).

    pool_idx   : candidate indices (whole pool or rerank window, caller's choice)
    pool_score : recall scores aligned with pool_idx
    rng        : numpy Generator — per-user deterministic random stream
    k          : how many to pick, default TOP_K_OUT = 10

    Pick k items at random FIRST, then present them sorted by descending
    relevance (argsort(-score, stable)) — same output semantics as rerank_ga,
    otherwise order-dependent metrics like HR@5 are not comparable across arms.
    """
    pool_idx = np.asarray(pool_idx)
    pool_score = np.asarray(pool_score)
    n = pool_idx.size
    rel_order = np.argsort(-pool_score, kind="stable")
    if n <= k:
        # Pool no larger than k: any random draw yields the whole pool, returned by descending relevance
        return pool_idx[rel_order], rel_order
    pick = rng.choice(n, size=k, replace=False)
    pick = pick[np.argsort(-pool_score[pick], kind="stable")]
    return pool_idx[pick], pick
