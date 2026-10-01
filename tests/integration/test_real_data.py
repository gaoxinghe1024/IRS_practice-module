"""Real-data checks (marker `real`): the activated MAL 2023 version, served from a throwaway DB copy.

Skipped when no real version is activated. Run with: make test-real
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import load_settings

pytestmark = pytest.mark.real


@pytest.fixture(scope="module")
def real_settings():
    s = load_settings("real", environ={})
    if not (s.artifacts_dir / "active.json").exists() or not s.db_path.exists():
        pytest.skip("no activated real dataset (see docs/operations.md)")
    return s


@pytest.fixture(scope="module")
def real_client(real_settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from evaluation.sandbox import sandbox_settings

    with sandbox_settings(real_settings) as sb, TestClient(create_app(sb), base_url="http://testserver") as c:
        yield c


def test_real_snapshot_pools_ids_and_missing_values(real_settings):
    from app.services.serving import load_state

    st = load_state(real_settings)
    snap = st.snap
    assert st.cap("catalog").status == "ready"
    keys = [str(k) for k in snap.keys]
    assert all(k.startswith("mal:") for k in keys) and len(set(keys)) == len(keys)      # MAL namespace only
    assert snap.recommendable.sum() <= snap.in_catalog.sum()
    rec = snap.recommendable & snap.in_catalog
    assert np.isfinite(snap.score[rec]).all()                   # recommendable => score known
    assert np.isnan(snap.score[snap.in_catalog]).any()          # unknown scores stay unknown (never 0)
    assert not (snap.score[snap.in_catalog] == 0).any()


def test_real_surfaces_end_to_end(real_client):
    c = real_client
    h = {"X-CSRF-Token": c.post("/api/v1/auth/register", json={"username": "real_check_user",
                                                              "password": "real check pw 1"}).json()["csrf_token"]}
    hits = c.get("/api/v1/anime", params={"q": "steins gate"}).json()["items"]
    assert "mal:9253" in [i["item_key"] for i in hits[:5]]

    sim = c.get("/api/v1/anime/mal:9253/similar").json()
    assert sim["meta"]["actual_k"] == 5 and all(i["reasons"] for i in sim["items"])
    assert "mal:9253" not in [i["item_key"] for i in sim["items"]]

    for key in ("mal:1535", "mal:19", "mal:9253", "mal:5114", "mal:16498", "mal:1575", "mal:31043", "mal:2904"):
        assert c.put(f"/api/v1/anime/{key}/rating", json={"rating": 9}, headers=h).status_code == 200
    home = c.post("/api/v1/recommendations/home", json={}, headers=h).json()
    assert home["mode"] == "warm" and len(home["items"]) == 10
    assert any(r["code"] == "icf_support" for i in home["items"] for r in i["reasons"])
    assert home["list_objective"]["fitness"] >= home["list_objective"]["greedy_baseline"]["fitness"] - 1e-9

    sid = c.post("/api/v1/chat/sessions", json={}, headers=h).json()["id"]
    ans = c.post(f"/api/v1/chat/sessions/{sid}/messages",
                 json={"message": "a comedy with at most 12 episodes and no romance"}, headers=h).json()
    assert ans["items"]
    for item in ans["items"]:
        detail = c.get(f"/api/v1/anime/{item['item_key']}").json()
        assert detail["episodes"] is not None and detail["episodes"] <= 12
        assert "Comedy" in detail["tags"].get("genre", []) and "Romance" not in detail["tags"].get("genre", [])

    plan = c.post("/api/v1/plans/preview", json={"budget_minutes": 900, "max_titles": 3}, headers=h).json()
    assert plan["feasibility"]["feasible"] and plan["total_minutes"] <= 900 and plan["items"]
