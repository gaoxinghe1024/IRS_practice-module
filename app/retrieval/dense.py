"""Dense retrieval over the versioned document vectors (normalised, exact cosine)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DenseHit:
    doc_id: str
    item_key: str
    kind: str
    score: float
    rank: int


class DenseIndex:
    def __init__(self, vectors: np.ndarray, doc_ids: list[str], item_keys: list[str], kinds: list[str],
                 encoder, info: dict):
        self.vectors = vectors
        self.doc_ids = doc_ids
        self.item_keys = np.asarray(item_keys, dtype=object)
        self.kinds = np.asarray(kinds, dtype=object)
        self.encoder = encoder
        self.info = info

    def search(self, query: str, eligible_keys: set[str] | None, top_n: int = 50,
               kinds: list[str] | None = None) -> list[DenseHit]:
        q = self.encoder.encode_queries([query])[0]
        s = self.vectors @ q
        mask = np.ones(len(s), dtype=bool)
        if eligible_keys is not None:
            mask &= np.fromiter((k in eligible_keys for k in self.item_keys), dtype=bool, count=len(s))
        if kinds:
            mask &= np.isin(self.kinds, kinds)
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        top = idx[np.lexsort((np.array(self.doc_ids, dtype=object)[idx], -s[idx]))[:top_n]]
        return [DenseHit(self.doc_ids[i], str(self.item_keys[i]), str(self.kinds[i]), float(s[i]), r + 1)
                for r, i in enumerate(top)]
