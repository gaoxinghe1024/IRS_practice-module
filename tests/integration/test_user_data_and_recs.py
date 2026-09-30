from tests.conftest import register


def _keys(client, n, **q):
    return [i["item_key"] for i in client.get("/api/v1/anime", params={"page_size": n, **q}).json()["items"]]


def test_rating_is_idempotent_and_bumps_profile_version(client):
    h, _ = register(client)
    key = _keys(client, 1)[0]
    v0 = client.get("/api/v1/me").json()["user"]["profile_version"]
    r1 = client.put(f"/api/v1/anime/{key}/rating", json={"rating": 9}, headers=h).json()
    r2 = client.put(f"/api/v1/anime/{key}/rating", json={"rating": 9}, headers=h).json()
    assert r1["profile_version"] == v0 + 1 and r2["profile_version"] == v0 + 1      # no-op repeat
    assert client.get("/api/v1/me/ratings").json()["total"] == 1
    assert client.put(f"/api/v1/anime/{key}/rating", json={"rating": 11}, headers=h).status_code == 422
    assert client.put(f"/api/v1/anime/{key}/rating", json={"rating": 7.5}, headers=h).status_code == 422
    assert client.delete(f"/api/v1/anime/{key}/rating", headers=h).json()["deleted"] is True
    assert client.get("/api/v1/me/ratings").json()["total"] == 0


def test_collections_membership_and_deletion_semantics(client):
    h, _ = register(client)
    a, b = _keys(client, 2)
    c1 = client.post("/api/v1/collections", json={"name": "One"}, headers=h).json()["id"]
    c2 = client.post("/api/v1/collections", json={"name": "Two"}, headers=h).json()["id"]
    assert client.post("/api/v1/collections", json={"name": "One"}, headers=h).status_code == 409
    assert client.put(f"/api/v1/collections/{c1}/items/{a}", headers=h).json()["added"] is True
    assert client.put(f"/api/v1/collections/{c1}/items/{a}", headers=h).json()["added"] is False
    client.put(f"/api/v1/collections/{c2}/items/{a}", headers=h)
    client.put(f"/api/v1/anime/{a}/rating", json={"rating": 8}, headers=h)
    client.delete(f"/api/v1/collections/{c1}/items/{a}", headers=h)
    assert [i["item_key"] for i in client.get(f"/api/v1/collections/{c2}").json()["items"]] == [a]
    assert client.get("/api/v1/me/ratings").json()["total"] == 1             # rating untouched
    assert client.delete(f"/api/v1/collections/{c2}", headers=h).status_code == 204
    assert client.get("/api/v1/me/ratings").json()["total"] == 1


def test_progress_cannot_exceed_known_episodes(client, snap):
    import numpy as np

    h, _ = register(client)
    i = int(np.flatnonzero(np.isfinite(snap.episodes) & snap.in_catalog)[0])
    key, eps = str(snap.keys[i]), int(snap.episodes[i])
    r = client.put(f"/api/v1/anime/{key}/progress", json={"status": "watching", "watched_episodes": eps + 1}, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "progress_exceeds_total"
    r = client.put(f"/api/v1/anime/{key}/progress", json={"status": "completed"}, headers=h).json()
    assert r["watched_episodes"] == eps


def test_similar_excludes_target_and_is_deterministic(client):
    key = _keys(client, 1, sort="popularity")[0]
    a = client.get(f"/api/v1/anime/{key}/similar").json()
    b = client.get(f"/api/v1/anime/{key}/similar").json()
    assert a["meta"]["actual_k"] <= 5 and key not in [i["item_key"] for i in a["items"]]
    assert [i["item_key"] for i in a["items"]] == [i["item_key"] for i in b["items"]]
    assert all(i["reasons"] for i in a["items"])


def test_home_excludes_rated_excluded_and_updates_with_profile(client):
    h, _ = register(client)
    keys = _keys(client, 12, sort="popularity")
    for k in keys[:9]:
        client.put(f"/api/v1/anime/{k}/rating", json={"rating": 9}, headers=h)
    home = client.post("/api/v1/recommendations/home", json={}, headers=h).json()
    got = [i["item_key"] for i in home["items"]]
    assert home["mode"] == "warm" and len(got) == len(set(got)) and not set(got) & set(keys[:9])
    victim = got[0]
    ex = client.post("/api/v1/feedback", json={"event_type": "not_interested", "item_key": victim}, headers=h).json()
    after = client.post("/api/v1/recommendations/home", json={}, headers=h).json()
    assert victim not in [i["item_key"] for i in after["items"]]
    sim = client.get(f"/api/v1/anime/{keys[0]}/similar", params={"k": 10}).json()
    assert victim not in [i["item_key"] for i in sim["items"]]
    client.delete(f"/api/v1/exclusions/{ex['exclusion_id']}", headers=h)
    again = client.post("/api/v1/recommendations/home", json={}, headers=h).json()
    assert again["meta"].get("cached") is not True                           # new profile_version -> recomputed


def test_negative_ratings_are_not_used_as_positive_seeds(client):
    h, _ = register(client)
    keys = _keys(client, 4, sort="popularity")
    client.put(f"/api/v1/anime/{keys[0]}/rating", json={"rating": 2}, headers=h)
    home = client.post("/api/v1/recommendations/home", json={}, headers=h).json()
    for item in home["items"]:
        for r in item["reasons"]:
            assert not (r["code"] == "icf_support" and r["params"]["rating"] <= 5)


def test_cold_start_user_gets_labelled_exploration(client):
    h, _ = register(client)
    home = client.post("/api/v1/recommendations/home", json={}, headers=h).json()
    assert home["mode"] == "cold" and home["items"]
    assert all(i["candidate_source"].startswith("popular") for i in home["items"])


def test_collection_rules_strict_empty_then_explicit_relaxation(client):
    h, _ = register(client)
    keys = _keys(client, 5, sort="popularity")
    cid = client.post("/api/v1/collections", json={"name": "Seeds"}, headers=h).json()["id"]
    for k in keys:
        client.put(f"/api/v1/collections/{cid}/items/{k}", headers=h)
    ok = client.post(f"/api/v1/collections/{cid}/recommendations", json={"max_episodes": 13}, headers=h).json()
    assert ok["status"] == "ok" and all(i["episodes"] is not None and i["episodes"] <= 13 for i in ok["items"])
    assert not set(keys) & {i["item_key"] for i in ok["items"]}              # saved titles excluded
    strict = client.post(f"/api/v1/collections/{cid}/recommendations", json={"max_episodes": 1, "min_score": 9.9},
                         headers=h).json()
    assert strict["status"] == "no_match" and strict["items"] == [] and strict["relaxation_options"]
    opt = strict["relaxation_options"][0]
    body = {"max_episodes": 1, "min_score": 9.9, **{k: v for k, v in opt["change"].items()
                                                   if k in ("max_episodes", "min_score")}}
    relaxed = client.post(f"/api/v1/collections/{cid}/recommendations",
                          json={**body, "relaxation_applied": {"rule_id": opt["rule_id"], "change": opt["change"]}},
                          headers=h).json()
    assert relaxed["relaxation_applied"]["rule_id"] == opt["rule_id"]


def test_user_data_and_plans_survive_an_application_restart(synthetic_env):
    from fastapi.testclient import TestClient

    from app.db.session import reset_engines
    from app.main import create_app

    name, pw = "restart_user", "restart password 1"
    with TestClient(create_app(synthetic_env), base_url="http://testserver") as c1:
        h = {"X-CSRF-Token": c1.post("/api/v1/auth/register", json={"username": name, "password": pw})
             .json()["csrf_token"]}
        key = c1.get("/api/v1/anime", params={"page_size": 1, "sort": "popularity"}).json()["items"][0]["item_key"]
        c1.put(f"/api/v1/anime/{key}/rating", json={"rating": 8}, headers=h)
        c1.put(f"/api/v1/anime/{key}/progress", json={"status": "watching", "watched_episodes": 1}, headers=h)
        cid = c1.post("/api/v1/collections", json={"name": "Keep"}, headers=h).json()["id"]
        c1.put(f"/api/v1/collections/{cid}/items/{key}", headers=h)
        pid = c1.post("/api/v1/plans", json={"budget_minutes": 600, "max_titles": 2}, headers=h).json()["id"]
    reset_engines()                                        # drop pooled connections: a cold start of the app
    with TestClient(create_app(synthetic_env), base_url="http://testserver") as c2:
        assert c2.post("/api/v1/auth/login", json={"username": name, "password": pw}).status_code == 200
        assert [(r["item_key"], r["rating"]) for r in c2.get("/api/v1/me/ratings").json()["items"]] == [(key, 8)]
        assert c2.get("/api/v1/me/progress").json()["items"][0]["watched_episodes"] == 1
        assert [i["item_key"] for i in c2.get(f"/api/v1/collections/{cid}").json()["items"]] == [key]
        plan = c2.get(f"/api/v1/plans/{pid}").json()
        assert plan["id"] == pid and plan["feasibility"]["feasible"]
