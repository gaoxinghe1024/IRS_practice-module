"""Leave-one-out cases for collection recommenders (training and evaluation).

For a collection C and hidden target h: the profile is rebuilt from C \\ {h} only,
so h's tags never leak into the profile. Positives and negatives are drawn from
the same eligible pool (the fixed rule layer: served, recommendable, not already
in the visible collection). Negatives are *sampled unobserved* items, not proven
dislikes. Splits are grouped by owner so one user's collections never appear
in both train and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.recommendation.constraints import ConstraintEngine, ConstraintSet
from app.recommendation.features import CatalogSnapshot
from app.recommendation.profiles import CollectionProfile, build_collection_profile


@dataclass
class LOOCase:
    case_id: str
    owner: str
    collection_id: str
    visible: list[int]
    target: int
    profile: CollectionProfile
    eligible: np.ndarray            # candidate pool for this case (target included when eligible)
    target_eligible: bool


def split_owner(owner: str, seed: int, fractions=(0.6, 0.2, 0.2)) -> str:
    from pipelines.processed_io import mix_hash

    h = float(mix_hash(np.array([abs(hash_str(owner))], dtype=np.int64), seed)[0])
    if h < fractions[0]:
        return "train"
    return "val" if h < fractions[0] + fractions[1] else "test"


def hash_str(s: str) -> int:
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def build_cases(snap: CatalogSnapshot, collections: list[dict], seed: int, max_targets: int = 3,
                min_size: int = 3) -> list[LOOCase]:
    rng = np.random.default_rng(seed)
    engine = ConstraintEngine(snap)
    base_mask = engine.evaluate(ConstraintSet()).mask
    cases = []
    for col in collections:
        items = [snap.index[k] for k in col["items"] if k in snap.index]
        items = list(dict.fromkeys(items))
        if len(items) < min_size:
            continue
        targets = rng.choice(items, size=min(max_targets, len(items)), replace=False)
        for t in targets:
            visible = [i for i in items if i != t]
            prof = build_collection_profile(snap, visible)
            m = base_mask.copy()
            m[visible] = False
            eligible = np.flatnonzero(m)
            cases.append(LOOCase(case_id=f"{col['collection_id']}:{int(t)}", owner=str(col["offline_user_id"]),
                                 collection_id=col["collection_id"], visible=visible, target=int(t), profile=prof,
                                 eligible=eligible, target_eligible=bool(m[t])))
    return cases


def sample_negatives(case: LOOCase, k: int, rng: np.random.Generator) -> np.ndarray:
    pool = case.eligible[case.eligible != case.target]
    if len(pool) == 0:
        return pool
    return rng.choice(pool, size=min(k, len(pool)), replace=False)
