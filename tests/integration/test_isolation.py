from tests.conftest import register


def _setup_owner(client):
    h, _ = register(client, "owner")
    key = client.get("/api/v1/anime", params={"page_size": 1}).json()["items"][0]["item_key"]
    cid = client.post("/api/v1/collections", json={"name": "Private"}, headers=h).json()["id"]
    client.put(f"/api/v1/collections/{cid}/items/{key}", headers=h)
    sid = client.post("/api/v1/chat/sessions", json={}, headers=h).json()["id"]
    client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "a comedy"}, headers=h)
    pid = client.post("/api/v1/plans", json={"budget_minutes": 600, "max_titles": 2}, headers=h).json()["id"]
    ex = client.post("/api/v1/feedback", json={"event_type": "not_interested", "item_key": key}, headers=h).json()
    return cid, sid, pid, ex["exclusion_id"]


def test_second_user_cannot_see_or_change_first_users_data(client):
    cid, sid, pid, exid = _setup_owner(client)
    client.cookies.clear()
    h2, _ = register(client, "intruder")
    for path in (f"/api/v1/collections/{cid}", f"/api/v1/chat/sessions/{sid}", f"/api/v1/plans/{pid}",
                 f"/api/v1/plans/{pid}/export?format=json"):
        r = client.get(path)
        assert r.status_code == 404, path                                  # existence is not revealed
    assert client.post(f"/api/v1/collections/{cid}/recommendations", json={}, headers=h2).status_code == 404
    assert client.delete(f"/api/v1/collections/{cid}", headers=h2).status_code == 404
    assert client.delete(f"/api/v1/exclusions/{exid}", headers=h2).status_code == 404
    r = client.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "hi"}, headers=h2)
    assert r.status_code == 404
    assert client.get("/api/v1/collections").json()["items"] == []
    assert client.get("/api/v1/me/history").json() == {"chats": [], "plans": []}


def test_admin_endpoints_require_admin(client):
    h, _ = register(client)
    assert client.post("/api/v1/admin/reload", headers=h).status_code == 403
    assert client.get("/api/v1/admin/runs/whatever").status_code == 403
    assert client.get("/dev").status_code == 404                          # hidden for non-admins
