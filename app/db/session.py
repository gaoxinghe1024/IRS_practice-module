"""Engine / session management for the SQLite business database.

Every connection enables foreign keys, WAL journaling and a busy timeout.
Write transactions are short; ``write_retry`` retries a bounded number of
times when SQLite reports the database as locked.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import PROJECT_ROOT, get_settings

T = TypeVar("T")


def _on_connect(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


@lru_cache(maxsize=8)
def get_engine(db_url: str | None = None) -> Engine:
    url = db_url or get_settings().db_url
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 5}, pool_pre_ping=True)
    event.listen(engine, "connect", _on_connect)
    return engine


@lru_cache(maxsize=8)
def _factory(db_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(db_url), expire_on_commit=False, autoflush=False)


def session_factory(db_url: str | None = None) -> sessionmaker[Session]:
    return _factory(db_url or get_settings().db_url)


@contextmanager
def session_scope(db_url: str | None = None) -> Iterator[Session]:
    session = session_factory(db_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def write_retry(fn: Callable[[], T], attempts: int = 4, base_delay: float = 0.05) -> T:
    for i in range(attempts):
        try:
            return fn()
        except OperationalError as exc:  # database is locked
            if "locked" not in str(exc).lower() or i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError("unreachable")


def run_migrations(db_url: str | None = None) -> None:
    """Apply Alembic migrations up to head (idempotent)."""
    from alembic import command
    from alembic.config import Config

    url = db_url or get_settings().db_url
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def reset_engines() -> None:
    for fn in (get_engine, _factory):
        fn.cache_clear()
