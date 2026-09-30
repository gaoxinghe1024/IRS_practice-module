"""Ratings, collections, watch progress, preferences, exclusions and feedback.

Every write that can change recommendations increments users.profile_version,
so cache keys built from it are invalidated immediately. Repeated writes are
idempotent updates (no duplicate business rows). Ownership is always checked
on the server; another user's collection is reported as not found.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.errors import Conflict, NotFound, ValidationFailed
from app.core.text import norm
from app.db.models import (
    Anime,
    Collection,
    CollectionItem,
    Exclusion,
    FeedbackEvent,
    Preference,
    Rating,
    Tag,
    User,
    WatchProgress,
    utcnow,
)
from app.recommendation.features import CatalogSnapshot
from app.recommendation.profiles import UserState

PROGRESS_STATUSES = ("plan_to_watch", "watching", "completed", "on_hold", "dropped")
FEEDBACK_TYPES = ("click", "save", "unsave", "rate", "not_interested", "undo_not_interested", "helpful",
                  "unhelpful", "plan_saved", "relaxation_applied")
LENGTH_PREFS = ("any", "short", "medium", "long")


def bump(user: User) -> None:
    user.profile_version = (user.profile_version or 0) + 1


def served_anime(db: Session, item_key: str) -> Anime:
    a = db.get(Anime, item_key)
    if a is None or not a.in_catalog or a.is_explicit:
        raise NotFound("Anime not found.", code="anime_not_found")
    return a


# --------------------------------------------------------------------------- ratings


def put_rating(db: Session, user: User, item_key: str, rating: int) -> Rating:
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 10:
        raise ValidationFailed("Rating must be an integer from 1 to 10.", code="invalid_rating")
    served_anime(db, item_key)
    row = db.scalar(select(Rating).where(Rating.user_id == user.id, Rating.item_key == item_key))
    now = utcnow()
    if row is None:
        row = Rating(user_id=user.id, item_key=item_key, rating=rating, created_at=now, updated_at=now)
        db.add(row)
        bump(user)
    elif row.rating != rating:
        row.rating, row.updated_at = rating, now
        bump(user)
    db.flush()
    return row


def delete_rating(db: Session, user: User, item_key: str) -> bool:
    res = db.execute(delete(Rating).where(Rating.user_id == user.id, Rating.item_key == item_key))
    if res.rowcount:
        bump(user)
    return bool(res.rowcount)


def list_ratings(db: Session, user: User, limit: int = 50, offset: int = 0) -> tuple[list, int]:
    q = (select(Rating, Anime).join(Anime, Anime.item_key == Rating.item_key).where(Rating.user_id == user.id)
         .order_by(Rating.updated_at.desc(), Rating.id.desc()))
    total = db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id))
    return db.execute(q.limit(limit).offset(offset)).all(), int(total or 0)


# --------------------------------------------------------------------------- collections


def _owned_collection(db: Session, user: User, collection_id: int) -> Collection:
    c = db.get(Collection, collection_id)
    if c is None or c.user_id != user.id:
        raise NotFound("Collection not found.", code="collection_not_found")
    return c


def list_collections(db: Session, user: User) -> list[dict]:
    rows = db.execute(select(Collection, func.count(CollectionItem.item_key))
                      .outerjoin(CollectionItem, CollectionItem.collection_id == Collection.id)
                      .where(Collection.user_id == user.id).group_by(Collection.id)
                      .order_by(Collection.created_at, Collection.id)).all()
    out = []
    for c, n in rows:
        cover = db.scalar(select(Anime.cover_url).join(CollectionItem, CollectionItem.item_key == Anime.item_key)
                          .where(CollectionItem.collection_id == c.id, Anime.cover_url.is_not(None))
                          .order_by(CollectionItem.added_at).limit(1))
        out.append({"id": c.id, "name": c.name, "description": c.description, "item_count": int(n),
                    "cover_url": cover, "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat()})
    return out


def _clean_name(name: str) -> str:
    name = " ".join((name or "").split())
    if not 1 <= len(name) <= 60:
        raise ValidationFailed("Collection name must be 1-60 characters.", code="invalid_collection_name",
                               details={"field": "name"})
    return name


def create_collection(db: Session, user: User, name: str, description: str = "") -> Collection:
    name = _clean_name(name)
    if db.scalar(select(Collection.id).where(Collection.user_id == user.id, Collection.name == name)):
        raise Conflict("You already have a collection with that name.", code="collection_name_taken")
    c = Collection(user_id=user.id, name=name, description=(description or "")[:300])
    db.add(c)
    db.flush()
    return c


def get_collection(db: Session, user: User, collection_id: int) -> tuple[Collection, list]:
    c = _owned_collection(db, user, collection_id)
    items = db.execute(select(CollectionItem, Anime).join(Anime, Anime.item_key == CollectionItem.item_key)
                       .where(CollectionItem.collection_id == c.id)
                       .order_by(CollectionItem.added_at, CollectionItem.item_key)).all()
    return c, items


def update_collection(db: Session, user: User, collection_id: int, name: str | None, description: str | None
                      ) -> Collection:
    c = _owned_collection(db, user, collection_id)
    if name is not None:
        name = _clean_name(name)
        clash = db.scalar(select(Collection.id).where(Collection.user_id == user.id, Collection.name == name,
                                                      Collection.id != c.id))
        if clash:
            raise Conflict("You already have a collection with that name.", code="collection_name_taken")
        c.name = name
    if description is not None:
        c.description = description[:300]
    c.updated_at = utcnow()
    return c


def delete_collection(db: Session, user: User, collection_id: int) -> None:
    c = _owned_collection(db, user, collection_id)
    db.delete(c)          # removes its items only; ratings and other collections are untouched
    bump(user)


def add_collection_item(db: Session, user: User, collection_id: int, item_key: str) -> bool:
    c = _owned_collection(db, user, collection_id)
    served_anime(db, item_key)
    if db.get(CollectionItem, (c.id, item_key)) is not None:
        return False
    db.add(CollectionItem(collection_id=c.id, item_key=item_key))
    c.updated_at = utcnow()
    bump(user)
    db.flush()
    return True


def remove_collection_item(db: Session, user: User, collection_id: int, item_key: str) -> bool:
    c = _owned_collection(db, user, collection_id)
    res = db.execute(delete(CollectionItem).where(CollectionItem.collection_id == c.id,
                                                  CollectionItem.item_key == item_key))
    if res.rowcount:
        c.updated_at = utcnow()
        bump(user)
    return bool(res.rowcount)


def collections_containing(db: Session, user: User, item_key: str) -> list[int]:
    return list(db.scalars(select(CollectionItem.collection_id).join(Collection)
                           .where(Collection.user_id == user.id, CollectionItem.item_key == item_key)))


# --------------------------------------------------------------------------- progress


def put_progress(db: Session, user: User, item_key: str, status: str, watched_episodes: int | None) -> WatchProgress:
    a = served_anime(db, item_key)
    if status not in PROGRESS_STATUSES:
        raise ValidationFailed(f"Status must be one of {', '.join(PROGRESS_STATUSES)}.", code="invalid_status")
    if watched_episodes is None:
        watched_episodes = a.episodes if (status == "completed" and a.episodes) else 0
    if watched_episodes < 0:
        raise ValidationFailed("Watched episodes cannot be negative.", code="invalid_progress")
    if a.episodes is not None and watched_episodes > a.episodes:
        raise ValidationFailed(f"This title has {a.episodes} episodes.", code="progress_exceeds_total",
                               details={"total_episodes": a.episodes})
    row = db.get(WatchProgress, (user.id, item_key))
    if row is None:
        row = WatchProgress(user_id=user.id, item_key=item_key, status=status, watched_episodes=watched_episodes)
        db.add(row)
    else:
        row.status, row.watched_episodes = status, watched_episodes
    row.updated_at = utcnow()
    bump(user)
    db.flush()
    return row


def delete_progress(db: Session, user: User, item_key: str) -> bool:
    res = db.execute(delete(WatchProgress).where(WatchProgress.user_id == user.id, WatchProgress.item_key == item_key))
    if res.rowcount:
        bump(user)
    return bool(res.rowcount)


# --------------------------------------------------------------------------- preferences / exclusions


def resolve_tag(db: Session, namespace: str, value: str) -> Tag:
    tag = db.scalar(select(Tag).where(Tag.namespace == namespace, Tag.normalized == norm(value)))
    if tag is None:
        raise ValidationFailed(f"Unknown {namespace} tag: {value}", code="unknown_tag")
    return tag


def get_preferences(db: Session, user: User) -> Preference:
    pref = db.get(Preference, user.id)
    if pref is None:
        pref = Preference(user_id=user.id)
        db.add(pref)
        db.flush()
    return pref


def update_preferences(db: Session, user: User, liked_tags: list[tuple[str, str]] | None,
                       length_preference: str | None, hide_watched_in_similar: bool | None) -> Preference:
    pref = get_preferences(db, user)
    if liked_tags is not None:
        pref.liked_tag_ids = sorted({resolve_tag(db, ns, v).id for ns, v in liked_tags})
    if length_preference is not None:
        if length_preference not in LENGTH_PREFS:
            raise ValidationFailed("Invalid length preference.", code="invalid_length_preference")
        pref.length_preference = length_preference
    if hide_watched_in_similar is not None:
        pref.hide_watched_in_similar = bool(hide_watched_in_similar)
    pref.updated_at = utcnow()
    bump(user)
    return pref


def add_exclusion(db: Session, user: User, kind: str, item_key: str | None = None, tag: tuple[str, str] | None = None,
                  source: str = "not_interested") -> Exclusion:
    if kind == "item":
        served_anime(db, item_key or "")
        existing = db.scalar(select(Exclusion).where(Exclusion.user_id == user.id, Exclusion.kind == "item",
                                                     Exclusion.item_key == item_key, Exclusion.revoked_at.is_(None)))
        if existing:
            return existing
        ex = Exclusion(user_id=user.id, kind="item", item_key=item_key, source=source)
    elif kind == "tag" and tag:
        t = resolve_tag(db, *tag)
        existing = db.scalar(select(Exclusion).where(Exclusion.user_id == user.id, Exclusion.kind == "tag",
                                                     Exclusion.tag_id == t.id, Exclusion.revoked_at.is_(None)))
        if existing:
            return existing
        ex = Exclusion(user_id=user.id, kind="tag", tag_id=t.id, source=source)
    else:
        raise ValidationFailed("Exclusion must be an item or a tag.", code="invalid_exclusion")
    db.add(ex)
    bump(user)
    db.flush()
    return ex


def revoke_exclusion(db: Session, user: User, exclusion_id: int) -> Exclusion:
    ex = db.get(Exclusion, exclusion_id)
    if ex is None or ex.user_id != user.id:
        raise NotFound("Exclusion not found.", code="exclusion_not_found")
    if ex.revoked_at is None:
        ex.revoked_at = utcnow()
        bump(user)
    return ex


def list_exclusions(db: Session, user: User) -> list[dict]:
    rows = db.execute(select(Exclusion, Anime.title, Tag.namespace, Tag.value)
                      .outerjoin(Anime, Anime.item_key == Exclusion.item_key)
                      .outerjoin(Tag, Tag.id == Exclusion.tag_id)
                      .where(Exclusion.user_id == user.id, Exclusion.revoked_at.is_(None))
                      .order_by(Exclusion.created_at.desc())).all()
    return [{"id": e.id, "kind": e.kind, "item_key": e.item_key, "title": title,
             "tag": {"namespace": ns, "value": v} if e.kind == "tag" else None, "source": e.source,
             "created_at": e.created_at.isoformat()} for e, title, ns, v in rows]


def record_feedback(db: Session, user: User, event_type: str, item_key: str | None, request_id: str | None,
                    surface: str | None, algorithm: str | None, value: str | None, context: dict | None
                    ) -> FeedbackEvent:
    if event_type not in FEEDBACK_TYPES:
        raise ValidationFailed("Unknown feedback type.", code="invalid_feedback_type")
    if item_key:
        served_anime(db, item_key)
    ev = FeedbackEvent(user_id=user.id, request_id=request_id, item_key=item_key, event_type=event_type,
                       value=value, is_explicit=event_type != "click", surface=surface, algorithm=algorithm,
                       context=context or {})
    db.add(ev)
    db.flush()
    return ev


# --------------------------------------------------------------------------- state for recommenders


def load_user_state(db: Session, user: User | None, snap: CatalogSnapshot) -> UserState:
    if user is None:
        return UserState(user_id=None, profile_version=0)
    st = UserState(user_id=user.id, profile_version=user.profile_version)
    for key, r in db.execute(select(Rating.item_key, Rating.rating).where(Rating.user_id == user.id)):
        i = snap.index.get(key)
        if i is not None:
            st.ratings[i] = float(r)
    for cid, key in db.execute(select(CollectionItem.collection_id, CollectionItem.item_key).join(Collection)
                               .where(Collection.user_id == user.id)):
        i = snap.index.get(key)
        if i is not None:
            st.collections.setdefault(cid, []).append(i)
    for key, status, eps in db.execute(select(WatchProgress.item_key, WatchProgress.status,
                                              WatchProgress.watched_episodes).where(WatchProgress.user_id == user.id)):
        i = snap.index.get(key)
        if i is not None:
            st.progress[i] = (status, int(eps))
    for kind, key, ns, val in db.execute(select(Exclusion.kind, Exclusion.item_key, Tag.namespace, Tag.value)
                                         .outerjoin(Tag, Tag.id == Exclusion.tag_id)
                                         .where(Exclusion.user_id == user.id, Exclusion.revoked_at.is_(None))):
        if kind == "item" and key in snap.index:
            st.excluded_items.add(snap.index[key])
        elif kind == "tag" and ns:
            st.excluded_tags.append((ns, val))
    pref = db.get(Preference, user.id)
    if pref and pref.liked_tag_ids:
        for ns, val in db.execute(select(Tag.namespace, Tag.value).where(Tag.id.in_(pref.liked_tag_ids))):
            st.liked_tags.append((ns, val))
    return st
