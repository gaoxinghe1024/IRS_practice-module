"""Structured JSON logging with request ids. Never logs passwords, cookies or keys."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_REDACT_KEYS = {"password", "token", "cookie", "authorization", "api_key", "x-csrf-token", "session"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: ("[redacted]" if k.lower() in _REDACT_KEYS else v) for k, v in extra.items()})
        if record.exc_info and record.levelno >= logging.ERROR:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str, ensure_ascii=False)


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("anilogic")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"anilogic.{name}")


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
