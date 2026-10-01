from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.core.text import split_keywords
from app.db.models import User
from app.schemas.requests import CollectionCreate, CollectionRecRequest, CollectionUpdate
from app.services import recommend, user_data
from app.services.serving import ServingState

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("")
def list_collections(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"items": user_data.list_collections(db, user)}


@router.post("", status_code=201)
def create(body: CollectionCreate, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    c = user_data.create_collection(db, user, body.name, body.description)
    return {"id": c.id, "name": c.name, "description": c.description, "item_count": 0}


@router.get("/{collection_id}")
def get(collection_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    c, rows = user_data.get_collection(db, user, collection_id)
    return {"id": c.id, "name": c.name, "description": c.description, "created_at": c.created_at.isoformat(),
            "items": [{"item_key": a.item_key, "title": a.title, "cover_url": a.cover_url, "type": a.type,
                       "episodes": a.episodes, "catalog_score": a.score, "added_at": ci.added_at.isoformat(),
                       "in_catalog": a.in_catalog} for ci, a in rows]}


@router.patch("/{collection_id}")
def update(collection_id: int, body: CollectionUpdate, db: Session = Depends(get_db),
           user: User = Depends(require_user)) -> dict:
    c = user_data.update_collection(db, user, collection_id, body.name, body.description)
    return {"id": c.id, "name": c.name, "description": c.description}


@router.delete("/{collection_id}", status_code=204)
def delete(collection_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> Response:
    user_data.delete_collection(db, user, collection_id)
    return Response(status_code=204)


@router.put("/{collection_id}/items/{item_key}")
def add_item(collection_id: int, item_key: str, db: Session = Depends(get_db),
             user: User = Depends(require_user)) -> dict:
    added = user_data.add_collection_item(db, user, collection_id, item_key)
    return {"collection_id": collection_id, "item_key": item_key, "added": added,
            "profile_version": user.profile_version}


@router.delete("/{collection_id}/items/{item_key}")
def remove_item(collection_id: int, item_key: str, db: Session = Depends(get_db),
                user: User = Depends(require_user)) -> dict:
    removed = user_data.remove_collection_item(db, user, collection_id, item_key)
    return {"collection_id": collection_id, "item_key": item_key, "removed": removed,
            "profile_version": user.profile_version}


@router.post("/{collection_id}/recommendations")
def recommendations(collection_id: int, body: CollectionRecRequest, db: Session = Depends(get_db),
                    user: User = Depends(require_user), state: ServingState = Depends(get_state)) -> dict:
    filters = recommend.CollectionFilters(keywords=split_keywords(body.keywords),
                                          include_tags=[(t.namespace, t.value) for t in body.include_tags],
                                          min_score=body.min_score, max_episodes=body.max_episodes,
                                          types=body.types or None)
    relax = body.relaxation_applied.model_dump() if body.relaxation_applied else None
    return recommend.collection(state, db, user, collection_id, filters, body.k, body.scorer, relax)
