from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, optional_user, require_user
from app.db.models import User
from app.schemas.requests import ProgressBody, RatingBody
from app.services import catalog, recommend, user_data
from app.services.serving import ServingState

router = APIRouter(prefix="/anime", tags=["anime"])


def _csv(v: str | None) -> list[str] | None:
    if not v:
        return None
    return [x.strip() for x in v.split(",") if x.strip()] or None


@router.get("")
def search(q: str = "", genres: str | None = None, exclude_genres: str | None = None, types: str | None = None,
           min_score: float | None = Query(None, ge=0, le=10), max_episodes: int | None = Query(None, ge=1),
           min_episodes: int | None = Query(None, ge=1), status: str | None = None,
           year_from: int | None = Query(None, ge=1900, le=2100), year_to: int | None = Query(None, ge=1900, le=2100),
           sort: str = "relevance", page: int = Query(1, ge=1), page_size: int = Query(24, ge=1, le=60),
           db: Session = Depends(get_db), state: ServingState = Depends(get_state)) -> dict:
    return catalog.search(state, db, q, _csv(genres), _csv(exclude_genres), _csv(types), min_score, max_episodes,
                          min_episodes, _csv(status), sort, page, page_size, year_from, year_to)


@router.get("/{item_key}")
def detail(item_key: str, db: Session = Depends(get_db), state: ServingState = Depends(get_state),
           user: User | None = Depends(optional_user)) -> dict:
    return catalog.detail(state, db, user, item_key)


@router.get("/{item_key}/similar")
def similar(item_key: str, k: int | None = Query(None, ge=1, le=10), scorer: str | None = None,
            hide_watched: bool | None = None, db: Session = Depends(get_db),
            state: ServingState = Depends(get_state), user: User | None = Depends(optional_user)) -> dict:
    return recommend.similar(state, db, user, item_key, k, scorer, hide_watched)


@router.get("/{item_key}/relations")
def relations(item_key: str, db: Session = Depends(get_db), state: ServingState = Depends(get_state)) -> dict:
    recommend.served_index(state, item_key)
    from app.graph.relations import start_years, watch_order

    return {"item_key": item_key, "relations": catalog.relations(state, db, item_key),
            "watch_order": watch_order(db, item_key, start_years(state.snap)),
            "capability": state.cap("relations").public()}


@router.put("/{item_key}/rating")
def put_rating(item_key: str, body: RatingBody, db: Session = Depends(get_db), user: User = Depends(require_user)
               ) -> dict:
    r = user_data.put_rating(db, user, item_key, body.rating)
    return {"item_key": item_key, "rating": r.rating, "profile_version": user.profile_version}


@router.delete("/{item_key}/rating")
def delete_rating(item_key: str, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    deleted = user_data.delete_rating(db, user, item_key)
    return {"item_key": item_key, "deleted": deleted, "profile_version": user.profile_version}


@router.put("/{item_key}/progress")
def put_progress(item_key: str, body: ProgressBody, db: Session = Depends(get_db),
                 user: User = Depends(require_user)) -> dict:
    p = user_data.put_progress(db, user, item_key, body.status, body.watched_episodes)
    return {"item_key": item_key, "status": p.status, "watched_episodes": p.watched_episodes,
            "profile_version": user.profile_version}


@router.delete("/{item_key}/progress")
def delete_progress(item_key: str, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"item_key": item_key, "deleted": user_data.delete_progress(db, user, item_key),
            "profile_version": user.profile_version}
