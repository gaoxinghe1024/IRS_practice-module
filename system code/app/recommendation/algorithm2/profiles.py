"""Canonical collection profile for algorithm 2, built only from the visible collection H.

    freq_ns(tag)   = (number of items in H carrying the tag) / |H|          (tags de-duplicated per item)
    known_ns       = number of items in H with at least one value in namespace ns
    median_eps     = median episode count over items of H with a known count (None if none)
    top tags       = the 10 most frequent namespaced tags over genre and theme ("genre:Action",
                     "theme:School"); ties broken by the namespaced label in ascending order

The hidden target never enters H, so it cannot influence the profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.recommendation.algorithm2.catalog import NAMESPACES, Alg2Catalog

PROFILE_VERSION = "alg2-profile-v1"
TOP_TAGS = 10
HIT_NAMESPACES = ("genre", "theme")


@dataclass
class Alg2Profile:
    history: np.ndarray                         # snapshot indices of H (sorted, unique)
    size: int
    freq: dict[str, np.ndarray]
    known: dict[str, int]
    median_eps: float | None
    top_tags: list[tuple[str, int]]             # (namespace, vocab column), best first
    top_vec: dict[str, np.ndarray] = field(default_factory=dict)
    mean_score: float | None = None

    @property
    def empty(self) -> bool:
        return self.size == 0

    def top_labels(self, cat: Alg2Catalog) -> list[str]:
        return [f"{ns}:{cat.vocab[ns][c]}" for ns, c in self.top_tags]

    def shares(self, cat: Alg2Catalog, ns: str, k: int = 5) -> list[dict]:
        f = self.freq.get(ns)
        if f is None or not f.any():
            return []
        order = sorted(np.flatnonzero(f > 0).tolist(), key=lambda c: (-f[c], cat.vocab[ns][c]))[:k]
        return [{"tag": cat.vocab[ns][c], "share": round(float(f[c]), 3)} for c in order]


def build_profile(cat: Alg2Catalog, history) -> Alg2Profile:
    h = np.unique(np.asarray(list(history), dtype=np.int64))
    size = len(h)
    freq: dict[str, np.ndarray] = {}
    known: dict[str, int] = {}
    for ns in NAMESPACES:
        X = cat.X[ns]
        if size:
            counts = np.asarray(X[h].sum(axis=0)).ravel()
            freq[ns] = counts / size
            known[ns] = int((cat.sizes[ns][h] > 0).sum())
        else:
            freq[ns] = np.zeros(X.shape[1])
            known[ns] = 0
    eps = cat.episodes[h] if size else np.zeros(0)
    eps = eps[np.isfinite(eps)]
    median_eps = float(np.median(eps)) if len(eps) else None
    ranked = []
    for ns in HIT_NAMESPACES:
        for c in np.flatnonzero(freq[ns] > 0):
            ranked.append((-freq[ns][c], f"{ns}:{cat.vocab[ns][c]}", ns, int(c)))
    ranked.sort()
    top = [(ns, c) for _, _, ns, c in ranked[:TOP_TAGS]]
    top_vec = {ns: np.zeros(cat.X[ns].shape[1], dtype=np.float32) for ns in HIT_NAMESPACES}
    for ns, c in top:
        top_vec[ns][c] = 1.0
    sc = cat.score[h] if size else np.zeros(0)
    sc = sc[np.isfinite(sc)]
    return Alg2Profile(history=h, size=size, freq=freq, known=known, median_eps=median_eps, top_tags=top,
                       top_vec=top_vec, mean_score=float(sc.mean()) if len(sc) else None)
