from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_state
from app.services.serving import ServingState

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def status(request: Request, state: ServingState = Depends(get_state)) -> dict:
    s = request.app.state.settings
    comps = state.manifest.get("components", {})
    return {"app": {"name": s.app.name, "tagline": s.app.tagline, "group": s.app.group_display},
            "dataset_mode": state.dataset_mode, "dataset_version": state.dataset_version,
            "capabilities": {k: v.public() for k, v in sorted(state.capabilities.items())},
            "models": {k: {"status": v.get("status"), "id": v.get("engine_id") or v.get("scorer_id")}
                       for k, v in comps.items()},
            "split_fingerprint": state.manifest.get("split", {}).get("fingerprint"),
            "loaded_at": state.loaded_at}


@router.get("/health")
def health(state: ServingState = Depends(get_state)) -> dict:
    return {"ok": state.ready, "dataset_version": state.dataset_version,
            "catalog": state.cap("catalog").status}
