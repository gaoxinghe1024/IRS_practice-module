import numpy as np

from tests.conftest import register


def _chat(client, h):
    return client.post("/api/v1/chat/sessions", json={}, headers=h).json()["id"]


def test_chat_constraints_are_enforced_and_refined(client, snap):
    h, _ = register(client)
    sid = _chat(client, h)
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages",
                    json={"message": "a comedy with at most 12 episodes and no horror"}, headers=h).json()
    assert r["answer_mode"] == "local_template" and r["items"]
    for it in r["items"]:
        i = snap.index[it["item_key"]]
        assert snap.episodes[i] <= 12 and "Comedy" in snap.tags["genre"].tags_of(i)
        assert "Horror" not in snap.tags["genre"].tags_of(i)
    ref = snap.titles[int(np.flatnonzero(snap.recommendable)[0])]
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": f"something like {ref} but shorter"},
                    headers=h).json()
    assert r["intent"] == "similar_to"
    r2 = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "at most 6 episodes"}, headers=h).json()
    assert r2["intent"] == "similar_to" and r2["refinement"]
    assert all(it["episodes"] is not None and it["episodes"] <= 6 for it in r2["items"])
    assert any(c["label"] == "<= 6 episodes" for c in r2["chips"])


def test_chat_never_invents_scene_episodes_or_relations(client, snap):
    h, _ = register(client)
    sid = _chat(client, h)
    movie = snap.titles[int(np.flatnonzero((snap.type == "Movie") & snap.in_catalog)[0])]
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages",
                    json={"message": f"which episode of {movie} has the rooftop confession?"}, headers=h).json()
    assert r["intent"] == "scene_lookup"
    for ev in r["evidence"]:
        if ev["kind"] == "scene" and ev["source_type"] != "scene_verified":
            assert "episode" not in r["answer"].lower() or "not recorded" in r["answer"].lower()
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "what is the weather today"},
                    headers=h).json()
    assert r["intent"] == "out_of_scope" and r["items"] == []


def test_llm_failure_falls_back_to_local_answer(client, monkeypatch):
    from app.chat import llm as llm_mod
    from app.services import chat as chat_mod

    class Failing:
        def complete(self, *a, **k):
            return llm_mod.LLMResult(ok=False, provider="stub", model="m", error="timeout")

    monkeypatch.setattr(chat_mod, "get_llm", lambda settings: Failing())
    h, _ = register(client)
    sid = _chat(client, h)
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "a mystery with at most 13 episodes"},
                    headers=h).json()
    assert r["answer_mode"] == "local_template" and r["items"]
    assert any("unavailable" in w for w in r["warnings"]) and r["meta"]["llm"][0]["error"] == "timeout"


def test_llm_answer_is_validated_before_display(client, monkeypatch):
    from app.chat import llm as llm_mod
    from app.services import chat as chat_mod

    class Good:
        def complete(self, system, user, json_schema=None, max_tokens=None):
            import json
            import re

            facts = json.loads(re.search(r"<facts>(.*?)</facts>", user, re.S).group(1))
            k = facts[0]["item_key"]
            return llm_mod.LLMResult(ok=True, provider="stub", model="m", usage={"input_tokens": 50, "output_tokens": 20},
                                     data={"answer": f"Try {facts[0]['title']}.", "item_keys": [k, "synthetic:99:999"],
                                           "claims": [], "unknowns": []})

    monkeypatch.setattr(chat_mod, "get_llm", lambda settings: Good())
    h, _ = register(client)
    sid = _chat(client, h)
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "a mystery with at most 13 episodes"},
                    headers=h).json()
    assert r["answer_mode"] == "llm" and r["meta"]["validation"]["dropped_items"] == ["synthetic:99:999"]
    assert all(not it["item_key"].startswith("synthetic:99") for it in r["items"])
    assert r["meta"]["llm"][0]["cost_status"] == "unknown"


def test_plan_budget_edits_revisions_and_export(client):
    h, _ = register(client)
    p = client.post("/api/v1/plans/preview", json={"budget_minutes": 900, "max_titles": 3}, headers=h).json()
    assert p["feasibility"]["feasible"] and p["total_minutes"] <= 900 and p["titles"] <= 3
    assert all(it["minutes"] > 0 and it["duration_evidence"]["basis"] for it in p["items"])
    saved = client.post("/api/v1/plans", json={"budget_minutes": 900, "max_titles": 3,
                                                "daily_slots": [{"label": "Sat", "minutes": 480},
                                                                {"label": "Sun", "minutes": 480}]}, headers=h).json()
    pid = saved["id"]
    assert saved["revision"] == 1
    first = saved["items"][0]["item_key"]
    edited = client.patch(f"/api/v1/plans/{pid}", json={"remove_items": [first]}, headers=h).json()
    assert edited["current_revision"] == 2 and first not in [i["item_key"] for i in edited["items"]]
    assert edited["feasibility"]["feasible"]
    for fmt in ("markdown", "html", "json"):          # a manually edited revision has no optimiser objective
        r = client.get(f"/api/v1/plans/{pid}/export", params={"format": fmt})
        assert r.status_code == 200, fmt
    assert "edited by you" in client.get(f"/api/v1/plans/{pid}/export", params={"format": "markdown"}).text
    tight = client.patch(f"/api/v1/plans/{pid}", json={"request": {"budget_minutes": 30, "max_titles": 3}},
                         headers=h).json()
    assert tight["current_revision"] == 3 and tight["total_minutes"] <= 30
    md = client.get(f"/api/v1/plans/{pid}/export", params={"format": "markdown"})
    assert md.status_code == 200 and "Budget" in md.text and "password" not in md.text.lower()
    html = client.get(f"/api/v1/plans/{pid}/export", params={"format": "html"})
    assert html.headers["content-type"].startswith("text/html") and "<table>" in html.text
    js = client.get(f"/api/v1/plans/{pid}/export", params={"format": "json"}).json()
    assert js["id"] == pid and js["revision"] == 3
    r = client.post("/api/v1/plans/preview", json={"budget_minutes": 600, "must_include": ["synthetic:1:999"]}, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "must_include_invalid"


def test_pages_render_and_protect_private_routes(client):
    for path in ("/", "/collections", "/chat", "/plan", "/profile", "/history", "/welcome"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login"), path
    for path in ("/discover", "/status", "/login", "/register"):
        assert client.get(path).status_code == 200, path
    register(client)
    for path in ("/", "/collections", "/chat", "/plan", "/profile", "/history"):
        r = client.get(path)
        assert r.status_code == 200 and 'id="i18n"' in r.text, path
    assert client.get("/no-such-page").status_code == 404
