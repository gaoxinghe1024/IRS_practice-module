"""Administrator-only endpoints: state reload, run traces, algorithm comparisons.

Normal users cannot trigger imports, activate models, load arbitrary model
paths or read other users' traces. Pipelines run from the CLI, not over HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_admin
from app.core.errors import NotFound
from app.db.models import RecommendationRun, User
from app.services import recommend
from app.services.serving import ServingState

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reload")
def reload(request: Request, _admin: User = Depends(require_admin)) -> dict:
    st = request.app.state.holder.reload()
    recommend.CACHE.clear()
    return {"dataset_version": st.dataset_version,
            "capabilities": {k: v.status for k, v in st.capabilities.items()}}


@router.get("/runs/{request_id}")
def run_trace(request_id: str, db: Session = Depends(get_db), _admin: User = Depends(require_admin)) -> dict:
    run = db.get(RecommendationRun, request_id)
    if run is None:
        raise NotFound("Run not found.")
    return {"request_id": run.request_id, "user_id": run.user_id, "surface": run.surface,
            "algorithm": run.algorithm, "model_version": run.model_version, "dataset_version": run.dataset_version,
            "params_fingerprint": run.params_fingerprint, "requested_k": run.requested_k, "actual_k": run.actual_k,
            "candidates": run.candidates, "results": run.results, "trace": run.trace,
            "created_at": run.created_at.isoformat()}


@router.get("/compare/similar/{item_key}")
def compare_similar(item_key: str, k: int = Query(5, ge=1, le=10), db: Session = Depends(get_db),
                    state: ServingState = Depends(get_state), _admin: User = Depends(require_admin)) -> dict:
    out = {}
    for scorer in sorted(state.similarity):
        res = recommend.similar(state, db, None, item_key, k, scorer, False)
        out[scorer] = [{"item_key": it["item_key"], "title": it["title"], "score": it["score"]} for it in res["items"]]
    return {"item_key": item_key, "scorers": out}


@router.get("/compare/home")
def compare_home(k: int = Query(10, ge=1, le=20), db: Session = Depends(get_db),
                 state: ServingState = Depends(get_state), admin: User = Depends(require_admin)) -> dict:
    """Engines x rerankers for the calling admin's own profile (never another user's)."""
    out = {}
    for engine in ("icf", "mf", "ncf"):
        for rr in ("greedy", "ga", "none"):
            res = recommend.home(state, db, admin, k, engine, rr)
            out[f"{engine}+{rr}"] = {"algorithm": res["meta"]["algorithm"], "fallback": res["meta"]["fallback"],
                                     "objective": res["list_objective"]["summary"],
                                     "items": [it["title"] for it in res["items"]]}
    return {"comparisons": out}
