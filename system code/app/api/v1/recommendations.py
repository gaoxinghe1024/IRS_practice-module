from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.db.models import User
from app.schemas.requests import ExclusionBody, FeedbackBody, HomeRequest
from app.services import recommend, user_data
from app.services.serving import ServingState

router = APIRouter(tags=["recommendations"])


@router.post("/recommendations/home")
def home(body: HomeRequest, db: Session = Depends(get_db), user: User = Depends(require_user),
         state: ServingState = Depends(get_state)) -> dict:
    return recommend.home(state, db, user, body.k, body.engine, body.reranker, body.candidate_k)


@router.post("/feedback", status_code=201)
def feedback(body: FeedbackBody, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    ev = user_data.record_feedback(db, user, body.event_type, body.item_key, body.request_id, body.surface,
                                   body.algorithm, body.value, body.context)
    out = {"id": ev.id, "event_type": ev.event_type}
    if body.event_type == "not_interested" and body.item_key:
        ex = user_data.add_exclusion(db, user, "item", item_key=body.item_key, source="not_interested")
        out["exclusion_id"] = ex.id
    out["profile_version"] = user.profile_version
    return out


@router.post("/exclusions", status_code=201)
def add_exclusion(body: ExclusionBody, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    tag = (body.tag.namespace, body.tag.value) if body.tag else None
    ex = user_data.add_exclusion(db, user, body.kind, item_key=body.item_key, tag=tag, source="explicit")
    return {"id": ex.id, "kind": ex.kind, "profile_version": user.profile_version}


@router.delete("/exclusions/{exclusion_id}")
def revoke_exclusion(exclusion_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    ex = user_data.revoke_exclusion(db, user, exclusion_id)
    user_data.record_feedback(db, user, "undo_not_interested", ex.item_key, None, "exclusions", None, None, {})
    return {"id": ex.id, "revoked": True, "profile_version": user.profile_version}


@router.get("/exclusions")
def list_exclusions(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"items": user_data.list_exclusions(db, user)}


def _no_content() -> Response:
    return Response(status_code=204)
