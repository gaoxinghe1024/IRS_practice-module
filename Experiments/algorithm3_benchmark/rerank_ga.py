# -*- coding: utf-8 -*-
"""
rerank_ga.py — Rerank layer 2: genetic algorithm (GA).

Fitness (slide convention):
    Fitness = 0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 freshness

Encoding: random keys. Each individual is a real-valued key vector of pool
size; the decoded Top-K subset is argsort(-keys)[:K]. Every key vector maps
to a valid subset (no repair step), so the whole population evaluates
vectorized — the key to running 20,000 users x 10 pool levels.

Randomness comes entirely from the caller-provided rng (candidate_pool.user_rng),
so GA results for a given user and pool level are exactly reproducible.
"""
import numpy as np

import config as C
from list_metrics import pool_similarity_matrix


def _fitness(pop, rel_norm, novelty, fresh, pers, pool_sim, mode):
    """Vectorized fitness evaluation over the whole decoded population.

    pop : (N, L) pool-position indices; the other vectors are pool-aligned.
    Returns (weighted total, per-component details).
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
        # SYSTEM_DEV_HANDOFF.md section 5
        total = 0.4 * rel + 0.3 * pe + 0.2 * div + 0.1 * nov
    else:
        # Slide convention: 0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 freshness
        total = (C.GA_W_RELEVANCE * rel + C.GA_W_DIVERSITY * div
                 + C.GA_W_NOVELTY * nov + C.GA_W_FRESH * fr)
    return total, dict(relevance=rel, diversity=div, novelty=nov,
                       fresh=fr, personalization=pe)


def rerank_ga(pool_idx, pool_score, tag_sets, novelty, fresh, pers, rng,
              mode=C.GA_FITNESS_MODE):
    """Run GA over the candidate pool. Returns (ordered items, positions within the pool, best fitness)."""
    n_pool = len(pool_idx)
    L = min(C.TOP_K_OUT, n_pool)
    if n_pool <= L:
        pos = np.arange(n_pool)
        return pool_idx[pos], pos, None

    # Normalize relevance to [0, 1] within the pool so the four weights are comparable
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
    # Seed one individual with the greedy solution: strictly descending keys
    keys[0] = np.linspace(1.0, 0.0, n_pool, dtype=np.float32)

    def decode(k):
        return np.argsort(-k, axis=1)[:, :L]

    pop = decode(keys)
    fit, _ = _fitness(pop, rel_norm, nov_p, fre_p, per_p, sim, mode)
    b = int(np.argmax(fit))
    best, best_fit = pop[b].copy(), float(fit[b])

    n_mut = max(1, int(C.GA_MUTATION * N))
    for _ in range(C.GA_GENERATIONS):
        # --- Tournament selection
        cand = rng.integers(0, N, size=(N, C.GA_TOURNAMENT))
        winner = cand[np.arange(N), np.argmax(fit[cand], axis=1)]
        new = keys[winner].copy()
        # --- Uniform crossover on the key vectors
        mask = rng.random((N, n_pool)) < C.GA_CROSSOVER
        partner = keys[rng.integers(0, N, size=N)]
        new = np.where(mask, new, partner).astype(np.float32)
        # --- Elite preservation
        keep = np.argsort(-fit)[:C.GA_ELITE]
        new[:C.GA_ELITE] = keys[keep]
        # --- Mutation: perturb one key of a subset of individuals
        rows = rng.choice(N, size=n_mut, replace=False)
        new[rows, rng.integers(0, n_pool, size=n_mut)] = rng.random(n_mut).astype(np.float32)
        keys = new

        pop = decode(keys)
        fit, _ = _fitness(pop, rel_norm, nov_p, fre_p, per_p, sim, mode)
        b = int(np.argmax(fit))
        if fit[b] > best_fit:
            best_fit, best = float(fit[b]), pop[b].copy()

    # Output order is still by descending relevance (keeps "earlier hit is better" semantics)
    order = best[np.argsort(-rel_norm[best], kind="stable")]
    return pool_idx[order], order, best_fit
