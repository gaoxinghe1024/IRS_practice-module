"""Hybrid retrieval over eligible titles + evidence loading.

Hard constraints are evaluated first (ConstraintEngine) and retrieval is
restricted to those titles; if a channel is unavailable the pipeline degrades
(dense -> lexical only) and says so, it never substitutes random results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeDocument
from app.retrieval.fusion import ItemResult, aggregate_items, rrf
from app.retrieval.lexical import LexicalRetriever


@dataclass
class RetrievalResult:
    items: list[ItemResult]
    channels: dict[str, int]
    degraded: list[str] = field(default_factory=list)


def hybrid_search(state, db: Session, query: str, eligible: set[str] | None, kinds: list[str] | None = None,
                  item_k: int = 5, evidence_per_item: int = 2) -> RetrievalResult:
    cfg = state.settings.retrieval
    n_lex, n_dense, k0 = int(cfg.get("lexical_top_n", 50)), int(cfg.get("dense_top_n", 50)), int(cfg.get("rrf_k0", 60))
    served = int((state.snap.in_catalog & ~state.snap.explicit).sum()) if state.snap is not None else 0
    # restrict inside SQL when the eligible set is selective; otherwise over-fetch and post-filter
    restrict = eligible is not None and len(eligible) < 0.5 * max(1, served)
    lex = LexicalRetriever(db.get_bind()).search(query, list(eligible) if restrict else None,
                                                 top_n=n_lex if restrict else n_lex * 4, kinds=kinds)
    if eligible is not None and not restrict:
        lex = [h for h in lex if h.item_key in eligible][:n_lex]
        for r, h in enumerate(lex, start=1):
            h.rank = r
    channels = {"lexical": lex}
    degraded = []
    if state.dense is not None:
        try:
            channels["dense"] = state.dense.search(query, eligible, top_n=n_dense, kinds=kinds)
        except Exception as exc:  # encoder failure -> lexical only (reported)
            degraded.append(f"dense retrieval failed ({type(exc).__name__}); lexical only")
    else:
        degraded.append("dense retrieval unavailable; lexical only")
    fused = rrf(channels, k0=k0)
    items = aggregate_items(fused, item_k, evidence_per_item)
    return RetrievalResult(items, {k: len(v) for k, v in channels.items()}, degraded)


def load_docs(db: Session, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    rows = db.scalars(select(KnowledgeDocument).where(KnowledgeDocument.doc_id.in_(doc_ids))).all()
    return {d.doc_id: {"doc_id": d.doc_id, "item_key": d.item_key, "kind": d.kind, "title": d.title, "body": d.body,
                       "source_type": d.source_type, "source_ref": d.source_ref, "source_url": d.source_url,
                       "spoiler_level": d.spoiler_level, "locator": d.locator or {}} for d in rows}


def item_docs(db: Session, item_key: str, kinds: list[str]) -> list[dict]:
    rows = db.scalars(select(KnowledgeDocument).where(KnowledgeDocument.item_key == item_key,
                                                      KnowledgeDocument.kind.in_(kinds))
                      .order_by(KnowledgeDocument.doc_id)).all()
    return [{"doc_id": d.doc_id, "item_key": d.item_key, "kind": d.kind, "title": d.title, "body": d.body,
             "source_type": d.source_type, "source_ref": d.source_ref, "source_url": d.source_url,
             "spoiler_level": d.spoiler_level, "locator": d.locator or {}} for d in rows]


def snippet(body: str, n: int = 240) -> str:
    body = " ".join(body.split())
    return body if len(body) <= n else body[:n].rsplit(" ", 1)[0] + " ..."
