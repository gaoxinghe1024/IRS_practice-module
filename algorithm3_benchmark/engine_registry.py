# -*- coding: utf-8 -*-
"""
engine_registry.py — unified loading and scoring interface for the three recall engines.

The engines differ internally (ICF is sparse-matrix, SVD/NCF are torch models), but
experiments only need two operations: load-and-fit (`load_engines`) and full-catalogue
scoring (`full_scores`), so experiment scripts never touch engine internals.

Engines:
    ICF  item-based CF over item-item similarity (deterministic, bit-identical retrains)
    SVD  Funk-SVD matrix factorisation, shared 32-dim latent space
    NCF  neural CF (MLP + negative sampling), 32-dim embeddings
ICF's determinism makes it the pipeline's regression anchor: after a rerun its ranks
must match history bit-for-bit, otherwise the environment changed.
"""
import time

import numpy as np

import config as C

ENGINES = ["ICF", "SVD", "NCF"]


def load_engines(names, R, n_items):
    """Fit and return {engine name: engine}. R is the training-ratings CSR."""
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
    """Full-catalogue scores (n_users, n_items) float32; ranking only, no clip."""
    if hasattr(eng, "score_items"):
        return np.asarray(eng.score_items(uu), dtype=np.float32)
    n = len(uu)
    uu_rep = np.repeat(np.asarray(uu, dtype=np.int64), eng.n_items)
    ii_rep = np.tile(np.arange(eng.n_items, dtype=np.int64), n)
    return eng.predict_pairs(uu_rep, ii_rep, clip=False).reshape(n, eng.n_items)
