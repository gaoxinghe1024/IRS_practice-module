from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.core.errors import CapabilityUnavailable
from app.db.models import Scene, User
from app.retrieval.evidence import hybrid_search, load_docs, snippet
from app.schemas.requests import SceneSearchBody
from app.services.recommend import card
from app.services.serving import ServingState

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.post("/search")
def search(body: SceneSearchBody, db: Session = Depends(get_db), user: User = Depends(require_user),
           state: ServingState = Depends(get_state)) -> dict:
    cap = state.cap("scenes")
    if cap.status in ("limited_by_data", "not_configured"):
        raise CapabilityUnavailable(cap.detail, code=cap.status)
    rr = hybrid_search(state, db, body.query, None, kinds=["scene"], item_k=body.k, evidence_per_item=2)
    docs = load_docs(db, [d.doc_id for it in rr.items for d in it.docs])
    scenes = {s.scene_id: s for s in db.scalars(select(Scene).where(Scene.scene_id.in_(
        [docs[d]["locator"].get("scene_id") for d in docs])))}
    out = []
    for it in rr.items:
        i = state.snap.index[it.item_key]
        matches = []
        for d in it.docs:
            doc = docs.get(d.doc_id)
            if not doc:
                continue
            sc = scenes.get(doc["locator"].get("scene_id"))
            verified = doc["source_type"] == "scene_verified"
            matches.append({"scene_id": doc["locator"].get("scene_id"), "snippet": snippet(doc["body"], 300),
                            "episode": doc["locator"].get("episode") if verified else None,
                            "timestamp_seconds": doc["locator"].get("timestamp_seconds") if verified else None,
                            "verification_status": sc.verification_status if sc else "unknown",
                            "spoiler_level": doc["spoiler_level"], "source_ref": doc["source_ref"],
                            "rrf": round(d.score, 5)})
        out.append({**card(state.snap, i), "matches": matches})
    return {"query": body.query, "items": out, "capability": cap.public(), "retrieval": rr.channels,
            "degraded": rr.degraded,
            "note": "Episode / timestamp are shown only for verified scene records; unknown stays unknown."}
