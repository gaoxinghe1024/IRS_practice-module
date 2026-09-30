"""FastAPI dependencies: database session, serving state, current user, CSRF, admin."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import Forbidden, Unauthorized
from app.core.security import constant_time_equals, token_hash
from app.db.models import User, UserSession
from app.db.session import session_factory
from app.services import auth
from app.services.serving import ServingState

SESSION_COOKIE = "anilogic_session"
CSRF_COOKIE = "anilogic_csrf"
CSRF_HEADER = "x-csrf-token"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    db = session_factory(request.app.state.settings.db_url)()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_state(request: Request) -> ServingState:
    return request.app.state.holder.get()


def _session(request: Request, db: Session) -> tuple[User, UserSession] | None:
    cached = getattr(request.state, "_auth", "unset")
    if cached != "unset":
        return cached
    found = auth.resolve(db, request.cookies.get(SESSION_COOKIE))
    request.state._auth = found
    return found


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    found = _session(request, db)
    if found and request.method in UNSAFE:
        _check_csrf(request, found[1])
    return found[0] if found else None


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    found = _session(request, db)
    if not found:
        raise Unauthorized("Please sign in.", code="not_authenticated")
    if request.method in UNSAFE:
        _check_csrf(request, found[1])
    return found[0]


def require_admin(user: User = Depends(require_user)) -> User:
    if not user.is_admin:
        raise Forbidden("Administrator access required.", code="admin_required")
    return user


def _check_csrf(request: Request, sess: UserSession) -> None:
    header = request.headers.get(CSRF_HEADER, "")
    if not header or not constant_time_equals(token_hash(header), sess.csrf_hash):
        raise Forbidden("Missing or invalid CSRF token. Reload the page and try again.", code="csrf_failed")
