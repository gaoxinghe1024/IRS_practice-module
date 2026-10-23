"""Application errors with stable codes and user-readable messages.

Business states such as "no titles match" are NOT errors; they are returned as
successful responses with warnings. Errors here map to 4xx/5xx responses.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, details: dict[str, Any] | None = None,
                 status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"


class CapabilityUnavailable(AppError):
    """A capability is not configured / lacks data. Never faked as success."""

    status_code = 503
    code = "capability_unavailable"


class ArtifactMismatch(AppError):
    status_code = 503
    code = "artifact_mismatch"
