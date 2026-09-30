"""Accounts and sessions.

Passwords: argon2id. Sessions: random 256-bit token in an HttpOnly cookie; only
its SHA-256 is stored. Each session also has a CSRF token (double-submit via a
readable cookie + X-CSRF-Token header, verified against the stored hash).
There is no default admin account; demo accounts exist only when the synthetic
seed explicitly creates them.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import Conflict, RateLimited, Unauthorized, ValidationFailed
from app.core.security import hash_password, new_token, token_hash, verify_password
from app.core.text import nfkc
from app.db.models import Preference, User, UserSession, utcnow

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
MIN_PASSWORD = 8
MAX_PASSWORD = 256


def normalize_username(username: str) -> str:
    return nfkc(username).strip().casefold()


@dataclass
class IssuedSession:
    user: User
    token: str
    csrf: str
    expires_at: datetime


class LoginThrottle:
    """At most `limit` failed attempts per username within `window` seconds."""

    def __init__(self, limit: int = 8, window: float = 600.0):
        self.limit, self.window = limit, window
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._fails.get(key, []) if now - t < self.window]
            self._fails[key] = recent
            if len(recent) >= self.limit:
                raise RateLimited("Too many failed sign-in attempts. Please wait a few minutes.")

    def fail(self, key: str) -> None:
        with self._lock:
            self._fails.setdefault(key, []).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


throttle = LoginThrottle()


def _issue(db: Session, user: User, ttl_hours: int) -> IssuedSession:
    token, csrf = new_token(), new_token(24)
    expires = utcnow() + timedelta(hours=ttl_hours)
    db.add(UserSession(user_id=user.id, token_hash=token_hash(token), csrf_hash=token_hash(csrf), expires_at=expires))
    db.flush()
    return IssuedSession(user, token, csrf, expires)


def register(db: Session, username: str, password: str, ttl_hours: int, origin: str = "registered") -> IssuedSession:
    display = nfkc(username or "").strip()
    if not USERNAME_RE.match(display):
        raise ValidationFailed("Username must be 3-32 characters: letters, digits, '_', '-' or '.'.",
                               code="invalid_username", details={"field": "username"})
    if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
        raise ValidationFailed(f"Password must be at least {MIN_PASSWORD} characters.", code="invalid_password",
                               details={"field": "password"})
    norm = normalize_username(display)
    if db.scalar(select(User.id).where(User.normalized_username == norm)) is not None:
        raise Conflict("That username is already taken.", code="username_taken", details={"field": "username"})
    user = User(normalized_username=norm, display_name=display, password_hash=hash_password(password), origin=origin)
    db.add(user)
    db.flush()
    db.add(Preference(user_id=user.id))
    return _issue(db, user, ttl_hours)


def login(db: Session, username: str, password: str, ttl_hours: int) -> IssuedSession:
    norm = normalize_username(username or "")
    throttle.check(norm)
    user = db.scalar(select(User).where(User.normalized_username == norm))
    ok, rehash = verify_password(password or "", user.password_hash) if user else (False, False)
    if not user or not ok:
        throttle.fail(norm)
        raise Unauthorized("Incorrect username or password.", code="invalid_credentials")
    throttle.reset(norm)
    if rehash:
        user.password_hash = hash_password(password)
    return _issue(db, user, ttl_hours)


def resolve(db: Session, token: str | None) -> tuple[User, UserSession] | None:
    if not token:
        return None
    sess = db.scalar(select(UserSession).where(UserSession.token_hash == token_hash(token)))
    if sess is None or sess.revoked_at is not None or sess.expires_at <= utcnow():
        return None
    user = db.get(User, sess.user_id)
    return (user, sess) if user else None


def logout(db: Session, token: str | None) -> None:
    found = resolve(db, token)
    if found:
        found[1].revoked_at = utcnow()
