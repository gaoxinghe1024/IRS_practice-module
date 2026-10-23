from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_state, require_user
from app.core.errors import CapabilityUnavailable
from app.db.models import User
from app.multimodal.image import validate_upload
from app.services.recommend import card
from app.services.serving import ServingState

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/identify")
async def identify(file: UploadFile = File(...), user: User = Depends(require_user),
                   state: ServingState = Depends(get_state)) -> dict:
    s = state.settings
    cap = state.cap("screenshot")
    limit = int(s.app.max_upload_bytes)
    data = await file.read(limit + 1)          # never read more than the limit + 1 byte
    img, info = validate_upload(data, limit, int(s.app.max_upload_pixels))
    if state.images is None:
        raise CapabilityUnavailable(cap.detail, code=cap.status)
    result = state.images.identify(img, top_n=int(s.images.get("top_n", 20)))
    for c in result["candidates"]:
        i = state.snap.index.get(c["item_key"])
        if i is not None:
            c.update({k: v for k, v in card(state.snap, i).items() if k != "item_key"})
    return {"upload": info, "capability": cap.public(), **result,
            "retained": bool(s.images.get("retain_uploads", False))}
