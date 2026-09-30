"""Catalogue browsing: search / filter / paginate, detail pages, tag lists, relations.

Browsing covers the whole served catalogue (including titles without a score);
only recommendation surfaces require the recommendable pool. A filter on a
field treats unknown values as not matching (unknown score never passes
"score >= 8"), and the UI shows unknown as unknown, never as 0.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationFailed
from app.db.models import (
    Anime,
    AnimeAlias,
    AnimeCharacter,
    AnimeTag,
    Character,
    Exclusion,
    KnowledgeDocument,
    Rating,
    Relation,
    Tag,
    User,
    WatchProgress,
)
from app.recommendation.constraints import ConstraintSet
from app.recommendation.features import stable_order
from app.retrieval.lexical import LexicalRetriever
from app.services.recommend import _require, card, served_index
from app.services.serving import ServingState
from app.services.user_data import collections_containing

SORTS = ("relevance", "score", "popularity", "year", "title")
PAGE_MAX = 60


def search(state: ServingState, db: Session, q: str = "", genres: list[str] | None = None,
           exclude_genres: list[str] | None = None, types: list[str] | None = None, min_score: float | None = None,
           max_episodes: int | None = None, min_episodes: int | None = None, status: list[str] | None = None,
           sort: str = "relevance", page: int = 1, page_size: int = 24) -> dict:
    snap = _require(state)
    if sort not in SORTS:
        raise ValidationFailed(f"sort must be one of {', '.join(SORTS)}", code="invalid_sort")
    if not 1 <= page_size <= PAGE_MAX or page < 1:
        raise ValidationFailed("Invalid page / page_size.", code="invalid_page")
    if min_score is not None and not 0 <= min_score <= 10:
        raise ValidationFailed("min_score must be between 0 and 10.", code="invalid_filter")
    q = (q or "").strip()[:200]
    cs = ConstraintSet(require_recommendable=False, min_score=min_score, max_episodes=max_episodes,
                       min_episodes=min_episodes, types=types or None, statuses=status or None,
                       include_tags=[("genre", g) for g in genres or []],
                       exclude_tags=[("genre", g) for g in exclude_genres or []])
    elig = state.constraints.evaluate(cs)
    mask = elig.mask
    match_info: dict[str, str] = {}
    if q:
        hits = LexicalRetriever(db.get_bind()).search_titles(q, top_n=500)
        order_keys = [k for k, _ in hits]
        if not order_keys:
            for m in state.titles.resolve(q, limit=10, fuzzy_cutoff=80.0):
                order_keys.append(m.item_key)
                match_info[m.item_key] = f"{m.match}: {m.alias}"
        ranked = [snap.index[k] for k in order_keys if k in snap.index and mask[snap.index[k]]]
        total = len(ranked)
        if sort != "relevance":
            ranked = _sorted(snap, np.array(ranked, dtype=np.int64), sort).tolist()
    else:
        idx = np.flatnonzero(mask)
        total = int(len(idx))
        ranked = _sorted(snap, idx, "popularity" if sort == "relevance" else sort).tolist()
    start = (page - 1) * page_size
    page_items = ranked[start:start + page_size]
    items = []
    for i in page_items:
        c = card(snap, int(i))
        if c["item_key"] in match_info:
            c["matched"] = match_info[c["item_key"]]
        items.append(c)
    return {"items": items, "total": total, "page": page, "page_size": page_size,
            "filters": cs.to_public() | {"q": q, "sort": sort},
            "rule_failures": elig.fail_counts, "dataset_mode": state.dataset_mode,
            "dataset_version": state.dataset_version}


def _sorted(snap, idx: np.ndarray, sort: str) -> np.ndarray:
    if len(idx) == 0:
        return idx
    keys = snap.keys[idx]
    if sort == "score":
        return idx[stable_order(snap.score[idx], keys)]
    if sort == "popularity":
        return idx[stable_order(np.nan_to_num(snap.members[idx], nan=-1), keys)]
    if sort == "year":
        return idx[stable_order(snap.year[idx], keys)]
    titles = np.array([snap.titles[i].casefold() for i in idx], dtype=object)
    return idx[np.lexsort((keys, titles))]


def detail(state: ServingState, db: Session, user: User | None, item_key: str) -> dict:
    snap = _require(state)
    i = served_index(state, item_key)
    a = db.get(Anime, item_key)
    tags: dict[str, list[str]] = {}
    for ns, v in db.execute(select(Tag.namespace, Tag.value).join(AnimeTag, AnimeTag.tag_id == Tag.id)
                            .where(AnimeTag.item_key == item_key).order_by(Tag.namespace, Tag.value)):
        tags.setdefault(ns, []).append(v)
    aliases = [{"alias": al.alias, "kind": al.kind} for al in
               db.scalars(select(AnimeAlias).where(AnimeAlias.item_key == item_key)) if al.kind != "title"]
    chars = [{"name": n, "role": r} for n, r in db.execute(
        select(Character.name, AnimeCharacter.role).join(AnimeCharacter, AnimeCharacter.character_id == Character.id)
        .where(AnimeCharacter.item_key == item_key).order_by(AnimeCharacter.role, Character.name)).all()[:24]]
    sources = [{"doc_id": d.doc_id, "kind": d.kind, "source_type": d.source_type, "source_ref": d.source_ref}
               for d in db.scalars(select(KnowledgeDocument).where(KnowledgeDocument.item_key == item_key)
                                   .order_by(KnowledgeDocument.doc_id))]
    out = {**card(snap, i), "synopsis": a.synopsis, "title_other": a.title_other,
           "duration_minutes": a.duration_minutes, "total_minutes": a.total_minutes, "duration_raw": a.duration_raw,
           "duration_basis": a.duration_basis, "status": a.status, "content_rating": a.content_rating,
           "source_material": a.source_material, "aired": a.aired_raw, "aired_from": a.aired_from,
           "aired_to": a.aired_to, "premiered": (f"{a.premiered_season} {a.premiered_year}"
                                                  if a.premiered_season else None),
           "scored_by": a.scored_by, "rank": a.rank, "popularity": a.popularity, "members": a.members,
           "favorites": a.favorites, "tags": tags, "aliases": aliases, "characters": chars,
           "relations": relations(state, db, item_key), "missing_fields": a.missing_fields or [],
           "provenance": a.provenance or {}, "is_recommendable": bool(a.is_recommendable),
           "sources": sources, "dataset_version": state.dataset_version, "dataset_mode": state.dataset_mode}
    if user is not None:
        r = db.scalar(select(Rating.rating).where(Rating.user_id == user.id, Rating.item_key == item_key))
        prog = db.get(WatchProgress, (user.id, item_key))
        ex = db.scalar(select(Exclusion.id).where(Exclusion.user_id == user.id, Exclusion.item_key == item_key,
                                                  Exclusion.revoked_at.is_(None)))
        out["user"] = {"rating": r, "collections": collections_containing(db, user, item_key),
                       "progress": {"status": prog.status, "watched_episodes": prog.watched_episodes} if prog else None,
                       "excluded_exclusion_id": ex}
    return out


def relations(state: ServingState, db: Session, item_key: str) -> list[dict]:
    rows = db.execute(select(Relation, Anime.title).outerjoin(Anime, Anime.item_key == Relation.to_item)
                      .where(Relation.from_item == item_key).order_by(Relation.relation_type, Relation.id)).all()
    out = []
    for r, to_title in rows:
        out.append({"relation_type": r.relation_type, "to_item": r.to_item,
                    "to_title": to_title or r.to_name, "to_entity_type": r.to_entity_type,
                    "linkable": r.to_item is not None, "status": r.status, "source_ref": r.source_ref})
    return out


def tag_list(state: ServingState, namespace: str) -> list[dict]:
    snap = _require(state)
    tm = snap.tags.get(namespace)
    if tm is None:
        raise ValidationFailed("Unknown tag namespace.", code="invalid_namespace")
    served = snap.in_catalog & ~snap.explicit
    counts = np.asarray(tm.X[served].sum(axis=0)).ravel()
    out = [{"value": v, "count": int(c)} for v, c in zip(tm.vocab, counts, strict=True) if c > 0]
    return sorted(out, key=lambda d: (-d["count"], d["value"]))
