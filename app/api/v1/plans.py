from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.db.models import User
from app.schemas.requests import PlanRequest, PlanUpdate
from app.services import plans
from app.services.serving import ServingState

router = APIRouter(prefix="/plans", tags=["plans"])


def _req(body: PlanRequest) -> dict:
    d = body.model_dump()
    d["daily_slots"] = [s for s in d.get("daily_slots", [])]
    return d


@router.post("/preview")
def preview(body: PlanRequest, db: Session = Depends(get_db), user: User = Depends(require_user),
            state: ServingState = Depends(get_state)) -> dict:
    """Compute a plan without saving it (saving is an explicit user action)."""
    return plans.compute(state, db, user, _req(body))


@router.get("")
def list_plans(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"items": plans.list_plans(db, user)}


@router.post("", status_code=201)
def create(body: PlanRequest, db: Session = Depends(get_db), user: User = Depends(require_user),
           state: ServingState = Depends(get_state)) -> dict:
    return plans.save(state, db, user, _req(body))


@router.get("/{plan_id}")
def get(plan_id: int, revision: int | None = Query(None, ge=1), db: Session = Depends(get_db),
        user: User = Depends(require_user), state: ServingState = Depends(get_state)) -> dict:
    return plans.get_plan(state, db, user, plan_id, revision)


@router.patch("/{plan_id}")
def update(plan_id: int, body: PlanUpdate, db: Session = Depends(get_db), user: User = Depends(require_user),
           state: ServingState = Depends(get_state)) -> dict:
    if body.request is not None:
        return plans.reoptimize(state, db, user, plan_id, _req(body.request))
    return plans.edit(state, db, user, plan_id, body.title, body.remove_items,
                      [e.model_dump() for e in body.edit_items], body.order)


@router.delete("/{plan_id}", status_code=204)
def delete(plan_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> Response:
    plans.delete(db, user, plan_id)
    return Response(status_code=204)


@router.post("/{plan_id}/reoptimize")
def reoptimize(plan_id: int, db: Session = Depends(get_db), user: User = Depends(require_user),
               state: ServingState = Depends(get_state)) -> dict:
    return plans.reoptimize(state, db, user, plan_id, None)


@router.get("/{plan_id}/export")
def export(plan_id: int, format: str = Query("markdown", pattern="^(json|markdown|html)$"),
           db: Session = Depends(get_db), user: User = Depends(require_user),
           state: ServingState = Depends(get_state)) -> Response:
    body, ctype, filename = plans.export(state, db, user, plan_id, format)
    return Response(content=body, media_type=ctype,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
