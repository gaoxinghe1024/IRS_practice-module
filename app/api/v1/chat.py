from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_state, require_user
from app.db.models import User
from app.schemas.requests import ChatMessageBody, ChatSessionCreate
from app.services import chat
from app.services.serving import ServingState

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return {"items": chat.list_sessions(db, user)}


@router.post("/sessions", status_code=201)
def create_session(body: ChatSessionCreate, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    s = chat.create_session(db, user, body.title)
    return {"id": s.id, "title": s.title}


@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> dict:
    return chat.get_session(db, user, session_id)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, db: Session = Depends(get_db), user: User = Depends(require_user)) -> Response:
    chat.delete_session(db, user, session_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/messages")
def post_message(session_id: int, body: ChatMessageBody, db: Session = Depends(get_db),
                 user: User = Depends(require_user), state: ServingState = Depends(get_state)) -> dict:
    return chat.post_message(state, db, user, session_id, body.message, body.save_preferences)
