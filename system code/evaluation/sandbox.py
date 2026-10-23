"""Throwaway copy of the activated business database for online-stack evaluations.

Chat, planner and system experiments exercise the real services (FTS, knowledge
documents, relations) but must not create evaluation users or chats in the
real business database; they run against a temporary copy instead.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.core.config import Settings
from app.db.session import reset_engines


@contextmanager
def sandbox_settings(settings: Settings):
    src = settings.db_path
    if not src.exists():
        raise FileNotFoundError(f"no activated database at {src}")
    tmp = Path(tempfile.mkdtemp(prefix="anilogic-eval-"))
    dst = tmp / "eval.sqlite3"
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:   # consistent copy even with WAL
        s.backup(d)
    sandbox = settings.model_copy(deep=True)
    sandbox._llm_api_key = settings.llm_api_key
    sandbox._db_path_override = str(dst)
    reset_engines()
    try:
        yield sandbox
    finally:
        reset_engines()
        shutil.rmtree(tmp, ignore_errors=True)


def make_user(settings: Settings, name: str):
    from app.db.session import session_scope
    from app.services import auth

    with session_scope(settings.db_url) as db:
        issued = auth.register(db, name, "evaluation-only-password", 1, origin="evaluation_sandbox")
        return issued.user.id
