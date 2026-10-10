"""Password hashing (argon2id), random session tokens and CSRF tokens.

Only hashes of session / CSRF tokens are stored. Legacy PBKDF2 hashes from the
old template (``salt$hexdigest``, 60000 iterations) can be verified once so a
migrated account can log in; they are re-hashed with argon2id on success.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

# argon2-cffi >= 23.1.0 用 InvalidHashError；旧版（如 21.3.0）只有 InvalidHash。
# 兼容两版，避免 import 在旧环境下直接失败。
try:
    from argon2.exceptions import InvalidHashError
except ImportError:  # pragma: no cover - 旧版 argon2
    from argon2.exceptions import InvalidHash as InvalidHashError

_hasher = PasswordHasher()  # argon2id with library defaults
LEGACY_PREFIX = "legacy-pbkdf2$"
_LEGACY_ITER = 60000


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """Return (ok, needs_rehash)."""
    if stored.startswith(LEGACY_PREFIX):
        try:
            salt, digest = stored[len(LEGACY_PREFIX):].split("$", 1)
        except ValueError:
            return False, False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _LEGACY_ITER).hex()
        ok = hmac.compare_digest(dk, digest)
        return ok, ok
    try:
        _hasher.verify(stored, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False, False
    return True, _hasher.check_needs_rehash(stored)


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
