from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.api.deps import CSRF_COOKIE, SESSION_COOKIE, get_db, get_settings_dep, optional_user
from app.core.config import Settings
from app.schemas.requests import Credentials
from app.services import auth

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_cookies(resp: Response, issued: auth.IssuedSession, settings: Settings) -> None:
    max_age = int(settings.app.session_ttl_hours * 3600)
    resp.set_cookie(SESSION_COOKIE, issued.token, max_age=max_age, httponly=True, samesite="lax",
                    secure=settings.app.secure_cookies, path="/")
    resp.set_cookie(CSRF_COOKIE, issued.csrf, max_age=max_age, httponly=False, samesite="strict",
                    secure=settings.app.secure_cookies, path="/")


def _user_payload(issued: auth.IssuedSession) -> dict:
    u = issued.user
    return {"user": {"id": u.id, "username": u.display_name, "onboarding_state": u.onboarding_state,
                     "is_admin": u.is_admin}, "csrf_token": issued.csrf}


@router.post("/register", status_code=201)
def register(body: Credentials, response: Response, db: Session = Depends(get_db),
             settings: Settings = Depends(get_settings_dep)) -> dict:
    issued = auth.register(db, body.username, body.password, settings.app.session_ttl_hours)
    db.commit()
    _set_cookies(response, issued, settings)
    return _user_payload(issued)


@router.post("/login")
def login(body: Credentials, response: Response, db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings_dep)) -> dict:
    issued = auth.login(db, body.username, body.password, settings.app.session_ttl_hours)
    db.commit()
    _set_cookies(response, issued, settings)
    return _user_payload(issued)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db),
           _user=Depends(optional_user)) -> dict:
    auth.logout(db, request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    return {"ok": True}
