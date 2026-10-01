"""Shared fixtures: a miniature synthetic environment built with the real pipelines.

The environment lives in a temporary directory (data, artifacts, business DB),
so tests never touch the developer's data. It uses the LSA text encoder so no
neural model download is needed; BGE is exercised by tests marked `ml`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import load_settings
from app.db.session import reset_engines

SMALL = {
    "ANILOGIC_SYNTHETIC__N_ANIME": "120",
    "ANILOGIC_SYNTHETIC__N_USERS": "300",
    "ANILOGIC_SYNTHETIC__TARGET_RATINGS": "6000",
    "ANILOGIC_SYNTHETIC__N_COLLECTORS": "90",
    "ANILOGIC_RETRIEVAL__EMBEDDING__PROVIDER": "lsa",
    "ANILOGIC_RETRIEVAL__EMBEDDING__LSA_DIM": "32",
    "ANILOGIC_RECOMMENDATION__MF__EPOCHS": "15",
    "ANILOGIC_RECOMMENDATION__NCF__EPOCHS": "3",
    "ANILOGIC_RECOMMENDATION__TWO_TOWER__EPOCHS": "2",
    "ANILOGIC_RECOMMENDATION__GA__TIME_BUDGET_MS": "2000",
    "ANILOGIC_PLANNER__TIME_BUDGET_MS": "3000",
    "ANILOGIC_LLM__PROVIDER": "none",
}


def env_for(tmp: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANILOGIC_")}
    env.update(SMALL)
    env.update({"ANILOGIC_PATHS__DATA_ROOT": str(tmp / "data"), "ANILOGIC_PATHS__ARTIFACTS_ROOT": str(tmp / "artifacts"),
                "ANILOGIC_PATHS__VAR_ROOT": str(tmp / "var"), "ANILOGIC_PATHS__EVALUATION_RUNS": str(tmp / "runs"),
                "ANILOGIC_DATASET_MODE": "synthetic"})
    return env


@pytest.fixture(scope="session")
def synthetic_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anilogic")
    env = env_for(tmp)
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        reset_engines()
        from pipelines.cli import main

        assert main(["build-all", "--mode", "synthetic"]) == 0
        settings = load_settings("synthetic")
        yield settings
    finally:
        os.environ.clear()
        os.environ.update(old)
        reset_engines()


@pytest.fixture()
def settings(synthetic_env):
    return synthetic_env


@pytest.fixture()
def client(synthetic_env):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services import recommend

    recommend.CACHE.clear()
    app = create_app(synthetic_env)
    with TestClient(app, base_url="http://testserver") as c:
        yield c


_counter = {"n": 0}


def register(client, prefix: str = "user") -> tuple[dict, dict]:
    """Register a fresh account; returns (csrf headers, user payload)."""
    _counter["n"] += 1
    r = client.post("/api/v1/auth/register", json={"username": f"{prefix}{_counter['n']}_{os.getpid()}",
                                                    "password": "correct horse battery"})
    assert r.status_code == 201, r.text
    return {"X-CSRF-Token": r.json()["csrf_token"]}, r.json()["user"]


@pytest.fixture()
def user(client):
    return register(client)


@pytest.fixture()
def snap(synthetic_env):
    from app.services.serving import load_state

    return load_state(synthetic_env).snap
