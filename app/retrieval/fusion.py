"""Reciprocal Rank Fusion and document -> title aggregation.

    RRF(doc) = sum over channels of 1 / (k0 + rank_channel(doc)),  k0 = 60 by default

Title level: a title's score is the best RRF of its documents (ties broken by
how many of its documents were retrieved, then item_key), so one title with
many chunks cannot fill the whole top-K list. Document-level and title-level
K are separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedDoc:
    doc_id: str
    item_key: str
    kind: str
    score: float
    channels: dict[str, int] = field(default_factory=dict)   # channel -> rank


@dataclass
class ItemResult:
    item_key: str
    score: float
    docs: list[FusedDoc]


def rrf(channels: dict[str, list], k0: int = 60) -> list[FusedDoc]:
    fused: dict[str, FusedDoc] = {}
    for name, hits in channels.items():
        for h in hits:
            d = fused.get(h.doc_id)
            if d is None:
                d = fused[h.doc_id] = FusedDoc(h.doc_id, h.item_key, h.kind, 0.0)
            d.score += 1.0 / (k0 + h.rank)
            d.channels[name] = h.rank
    return sorted(fused.values(), key=lambda d: (-d.score, d.doc_id))


def aggregate_items(docs: list[FusedDoc], item_k: int, evidence_per_item: int = 2) -> list[ItemResult]:
    by_item: dict[str, list[FusedDoc]] = {}
    for d in docs:
        by_item.setdefault(d.item_key, []).append(d)
    items = [ItemResult(k, max(x.score for x in v), sorted(v, key=lambda d: (-d.score, d.doc_id))[:evidence_per_item])
             for k, v in by_item.items()]
    items.sort(key=lambda it: (-it.score, -len(by_item[it.item_key]), it.item_key))
    return items[:item_k]
