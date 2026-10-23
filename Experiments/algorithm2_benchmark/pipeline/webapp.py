"""The real web app in-process (FastAPI TestClient) serving a frozen algorithm-2 run.

Runs against a throwaway copy of the activated real database (evaluation/sandbox.py), so evaluation
accounts and collections never reach the business database. Used by `benchmark` (API latency) and
`verify-integration` (API contract and offline/online agreement).
"""

from __future__ import annotations

from contextlib import contextmanager

from app.core.config import load_settings
from evaluation.sandbox import sandbox_settings

PASSWORD = "evaluation pw 1"


@contextmanager
def served_app(algorithm2_run: str | None, scorer: str | None = None):
    """Yield (app, client) for real mode with `recommendation.collection` pointing at a frozen run."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = load_settings("real", environ={})
    cc = settings.recommendation.setdefault("collection", {})
    cc["algorithm2_run"] = algorithm2_run
    if scorer:
        cc["scorer"] = scorer
    with sandbox_settings(settings) as sb:
        app = create_app(sb)
        with TestClient(app) as client:
            yield app, client


class Account:
    """One registered evaluation user on the sandbox app (cookie session + CSRF header)."""

    def __init__(self, client, username: str):
        r = client.post("/api/v1/auth/register", json={"username": username, "password": PASSWORD})
        if not r.is_success:
            raise RuntimeError(f"register {username}: HTTP {r.status_code} {r.text[:200]}")
        self.client = client
        self.h = {"X-CSRF-Token": r.json()["csrf_token"]}

    def collection(self, name: str, keys) -> int:
        r = self.client.post("/api/v1/collections", json={"name": name}, headers=self.h)
        if not r.is_success:
            raise RuntimeError(f"create collection: HTTP {r.status_code} {r.text[:200]}")
        cid = int(r.json()["id"])
        for key in keys:
            self.add(cid, key)
        return cid

    def add(self, cid: int, key: str) -> None:
        r = self.client.put(f"/api/v1/collections/{cid}/items/{key}", headers=self.h)
        if not r.is_success:
            raise RuntimeError(f"add {key} to collection {cid}: HTTP {r.status_code} {r.text[:200]}")

    def remove(self, cid: int, key: str) -> None:
        r = self.client.delete(f"/api/v1/collections/{cid}/items/{key}", headers=self.h)
        if not r.is_success:
            raise RuntimeError(f"remove {key} from collection {cid}: HTTP {r.status_code} {r.text[:200]}")

    def recommend(self, cid: int, body: dict):
        return self.client.post(f"/api/v1/collections/{cid}/recommendations", json=body, headers=self.h)


def rule_body(rule: str, keyword: str | None, r2: dict) -> dict | None:
    """The collection-page request for an offline rule scenario (None when R1/R3 have no keyword)."""
    body: dict = {}
    if rule in ("R1", "R3"):
        if not keyword:
            return None
        body["keywords"] = keyword
    if rule in ("R2", "R3"):
        body["min_score"] = float(r2["min_score"])
        body["max_episodes"] = int(r2["max_episodes"])
    return body
