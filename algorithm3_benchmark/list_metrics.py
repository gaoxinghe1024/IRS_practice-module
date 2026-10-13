# -*- coding: utf-8 -*-
"""
list_metrics.py — List-level metrics and feature preprocessing (basis of the GA fitness).

Fitness components beyond relevance are computed over the whole list:
diversity = 1 - mean pairwise Jaccard similarity (weight 0.3),
novelty = mean per-item novelty (weight 0.2),
freshness = share of items premiered in the last 5 years (weight 0.1).
Jaccard treats each item's (genres ∪ themes) as a set, so genre-clumped
lists score low on diversity — exactly what the GA pushes down.
"""
import numpy as np


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pool_similarity_matrix(pool_idx, tag_sets):
    """Pairwise Jaccard similarity matrix (n, n) inside the candidate pool."""
    n = len(pool_idx)
    M = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            v = jaccard(tag_sets[pool_idx[a]], tag_sets[pool_idx[b]])
            M[a, b] = M[b, a] = v
    return M


def list_metrics(items, tag_sets, novelty, fresh):
    """Diversity / novelty / freshness of one ordered list."""
    if len(items) == 0:
        return dict(diversity=0.0, novelty=0.0, fresh=0.0)
    L = len(items)
    sims = [jaccard(tag_sets[items[a]], tag_sets[items[b]])
            for a in range(L) for b in range(a + 1, L)]
    div = 1.0 - float(np.mean(sims)) if sims else 0.0
    return dict(diversity=div,
                novelty=float(np.mean([novelty[i] for i in items])),
                fresh=float(np.mean([fresh[i] for i in items])))


# ------------------------------------------------------------------ Feature preprocessing
def build_tag_sets(feats):
    """Deduplicated, sorted (genres ∪ themes) tuple per item, for Jaccard."""
    return [tuple(sorted(set(feats["genres"][i]) | set(feats["themes"][i])))
            for i in range(len(feats["genres"]))]


def fresh_flag(years, span=5):
    """Fresh = premiere year within the newest `span` years of the catalogue. Returns (0/1 vector, year threshold)."""
    ys = years[years > 0]
    if len(ys) == 0:
        return np.zeros(len(years), dtype=np.float32), 0
    thr = int(ys.max()) - (span - 1)
    return (years >= thr).astype(np.float32), thr
