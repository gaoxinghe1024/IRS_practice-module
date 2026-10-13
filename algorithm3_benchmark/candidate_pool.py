# -*- coding: utf-8 -*-
"""
candidate_pool.py — Candidate pool construction plus the per-user random streams for GA and arms.

Candidate pool (sampled negatives): each user's candidate set is
"1 hidden positive + (N-1) never-rated items", over the 10-level grid
N = 20/50/100/200/500/1000/2000/5000/10000/full catalogue.

GA stream: rng = default_rng([sampling_seed, pool_size, engine_id, user_id]),
independent and exactly reproducible per user, unaffected by chunking or
process splitting. Arm stream: adds an `arm` component (see arm_rng).
"""
import numpy as np

import config as C

# ------------------------------------------------------------------ Pool grid
GRID = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, None]
GRID_LABEL = [("full" if g is None else str(g)) for g in GRID]

SAMPLING_SEED_BASE = C.RANDOM_SEED + 7000    # 7042
SCORE_CHUNK = 1000                           # users per scoring batch (memory control)

# ------------------------------------------------------------------ Rerank window
RERANK_DEPTH = C.RECALL_POOL                 # 20: GA only sees the top-20 by recall score
FULL_GRID_KEY = 10 ** 7                      # sentinel grid key for the full-catalogue level

# Fixed: changing this order would shift the GA random streams
ENGINE_ID = {"ICF": 0, "SVD": 1, "NCF": 2}

# Random-arm ids (5th seed component of arm_rng)
ARM_ID = {"rand_window": 1, "rand_pool": 2}


def build_negative_pool(perm, rated_items, hidden):
    """Uniform random permutation of the user's non-excluded items (first N-1 are negatives).

    perm        : global random permutation for the seed (length n_items)
    rated_items : item indices the user rated in the training set
    hidden      : the single hidden item index of the user
    """
    excl = np.zeros(perm.size, dtype=bool)
    excl[rated_items] = True
    excl[hidden] = True
    return perm[~excl[perm]]


def user_rng(sampling_seed, grid_key, engine, user_id):
    """Per-user deterministic stream, keyed by the pool size itself (not the grid index)."""
    return np.random.default_rng([int(sampling_seed), int(grid_key),
                                  int(ENGINE_ID[engine]), int(user_id)])


def arm_rng(sampling_seed, grid_key, engine, user_id, arm):
    """Per-user random stream for the random arms (arm is a value from ARM_ID).

    Never append `arm` to the user_rng seed list: that changes the
    SeedSequence entropy and shifts ALL historical GA results, breaking
    every published number. The arms therefore use this separate stream.
    """
    return np.random.default_rng([int(sampling_seed), int(grid_key),
                                  int(ENGINE_ID[engine]), int(user_id), int(arm)])


def pool_permutation(seed, n_items):
    """Global item permutation for a given sampling seed."""
    return np.random.default_rng(seed).permutation(n_items)
