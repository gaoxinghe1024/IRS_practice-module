from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.core.errors import ValidationFailed
from app.db.models import Anime, ChatSession, Collection, PlanRevision, Rating, Tag, User, ViewingPlan, WatchProgress
from app.schemas.requests import PreferencesUpdate
from app.services import user_data
from app.services.serving import ServingState

router = APIRouter(tags=["me"])


def _prefs(db: Session, user: User) -> dict:
    pref = user_data.get_preferences(db, user)
    liked = []
    if pref.liked_tag_ids:
        liked = [{"namespace": ns, "value": v} for ns, v in
                 db.execute(select(Tag.namespace, Tag.value).where(Tag.id.in_(pref.liked_tag_ids))
                            .order_by(Tag.namespace, Tag.value))]
    return {"liked_tags": liked, "length_preference": pref.length_preference,
            "hide_watched_in_similar": pref.hide_watched_in_similar}


@router.get("/me")
def me(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    n_r = db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id))
    n_c = db.scalar(select(func.count()).select_from(Collection).where(Collection.user_id == user.id))
    return {"user": {"id": user.id, "username": user.display_name, "created_at": user.created_at.isoformat(),
                     "onboarding_state": user.onboarding_state, "is_admin": user.is_admin,
                     "profile_version": user.profile_version},
            "preferences": _prefs(db, user), "stats": {"ratings": n_r, "collections": n_c}}


@router.patch("/me/preferences")
def update_preferences(body: PreferencesUpdate, db: Session = Depends(get_db),
                       user: User = Depends(require_user)) -> dict:
    liked = [(t.namespace, t.value) for t in body.liked_tags] if body.liked_tags is not None else None
    user_data.update_preferences(db, user, liked, body.length_preference, body.hide_watched_in_similar)
    return {"preferences": _prefs(db, user), "profile_version": user.profile_version}


@router.post("/me/onboarding")
def onboarding(state: str = Query(pattern="^(done|skipped)$"), db: Session = Depends(get_db),
               user: User = Depends(require_user)) -> dict:
    user.onboarding_state = state
    return {"onboarding_state": state}


@router.get("/me/ratings")
def my_ratings(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db),
               user: User = Depends(require_user)) -> dict:
    rows, total = user_data.list_ratings(db, user, limit, offset)
    return {"total": total, "items": [{"item_key": a.item_key, "title": a.title, "cover_url": a.cover_url,
                                       "rating": r.rating, "updated_at": r.updated_at.isoformat(),
                                       "type": a.type, "episodes": a.episodes} for r, a in rows]}


@router.get("/me/progress")
def my_progress(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    rows = db.execute(select(WatchProgress, Anime).join(Anime, Anime.item_key == WatchProgress.item_key)
                      .where(WatchProgress.user_id == user.id).order_by(WatchProgress.updated_at.desc())).all()
    return {"items": [{"item_key": a.item_key, "title": a.title, "cover_url": a.cover_url, "status": p.status,
                       "watched_episodes": p.watched_episodes, "episodes": a.episodes,
                       "updated_at": p.updated_at.isoformat()} for p, a in rows]}


@router.get("/me/exclusions")
def my_exclusions(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"items": user_data.list_exclusions(db, user)}


@router.get("/me/history")
def my_history(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    chats = db.execute(select(ChatSession).where(ChatSession.user_id == user.id)
                       .order_by(ChatSession.updated_at.desc()).limit(100)).scalars().all()
    plans = db.execute(select(ViewingPlan).where(ViewingPlan.user_id == user.id)
                       .order_by(ViewingPlan.updated_at.desc()).limit(100)).scalars().all()
    plan_out = []
    for p in plans:
        rev = db.scalar(select(PlanRevision).where(PlanRevision.plan_id == p.id,
                                                   PlanRevision.revision == p.current_revision))
        plan_out.append({"id": p.id, "title": p.title, "revision": p.current_revision,
                         "updated_at": p.updated_at.isoformat(),
                         "summary": rev.summary if rev else {}})
    return {"chats": [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in chats],
            "plans": plan_out}


@router.get("/tags")
def tags(namespace: str = Query("genre"), state: ServingState = Depends(get_state)) -> dict:
    from app.services.catalog import tag_list

    if namespace not in ("genre", "theme", "demographic", "studio", "source"):
        raise ValidationFailed("Unknown namespace.", code="invalid_namespace")
    return {"namespace": namespace, "items": tag_list(state, namespace)}
