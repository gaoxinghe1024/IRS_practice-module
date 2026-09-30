from tests.conftest import register


def test_register_validation_and_duplicates(client):
    r = client.post("/api/v1/auth/register", json={"username": "ab", "password": "long enough pw"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_username"
    r = client.post("/api/v1/auth/register", json={"username": "shortpw_user", "password": "short"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_password"
    r = client.post("/api/v1/auth/register", json={"username": "Dup_User", "password": "long enough pw"})
    assert r.status_code == 201
    r = client.post("/api/v1/auth/register", json={"username": "dup_user", "password": "long enough pw"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "username_taken"   # case-insensitive


def test_session_cookie_flags_and_logout(client):
    r = client.post("/api/v1/auth/register", json={"username": "cookie_user", "password": "long enough pw"})
    cookies = r.headers.get_list("set-cookie")
    session = next(c for c in cookies if c.startswith("anilogic_session="))
    assert "HttpOnly" in session and "samesite=lax" in session.lower()
    csrf = r.json()["csrf_token"]
    assert client.get("/api/v1/me").status_code == 200
    assert client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    assert client.get("/api/v1/me").status_code == 401


def test_login_errors_are_generic_and_throttled(client):
    client.post("/api/v1/auth/register", json={"username": "throttle_user", "password": "long enough pw"})
    client.cookies.clear()
    r = client.post("/api/v1/auth/login", json={"username": "throttle_user", "password": "wrong password"})
    assert r.status_code == 401 and r.json()["error"]["message"] == "Incorrect username or password."
    r2 = client.post("/api/v1/auth/login", json={"username": "no_such_user_x", "password": "wrong password"})
    assert r2.json()["error"]["message"] == r.json()["error"]["message"]           # no user enumeration
    codes = [client.post("/api/v1/auth/login", json={"username": "throttle_user", "password": "bad"}).status_code
             for _ in range(10)]
    assert 429 in codes


def test_csrf_and_cross_origin_writes_are_rejected(client):
    headers, _ = register(client)
    key = client.get("/api/v1/anime", params={"page_size": 1}).json()["items"][0]["item_key"]
    assert client.put(f"/api/v1/anime/{key}/rating", json={"rating": 8}).status_code == 403
    r = client.put(f"/api/v1/anime/{key}/rating", json={"rating": 8}, headers={"X-CSRF-Token": "forged"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "csrf_failed"
    r = client.put(f"/api/v1/anime/{key}/rating", json={"rating": 8}, headers={**headers, "Origin": "https://evil.example"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "cross_origin_blocked"
    assert client.put(f"/api/v1/anime/{key}/rating", json={"rating": 8}, headers=headers).status_code == 200


def test_errors_have_stable_shape_without_internals(client):
    r = client.get("/api/v1/anime/mal:999999999")
    body = r.json()
    assert r.status_code == 404 and set(body["error"]) == {"code", "message", "details", "request_id"}
    assert "Traceback" not in r.text and r.headers["x-request-id"] == body["error"]["request_id"]
    r = client.get("/api/v1/anime", params={"sort": "bogus"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_sort"
    assert client.get("/api/v1/me").status_code == 401


def test_security_headers_on_pages(client):
    r = client.get("/discover")
    assert r.status_code == 200
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert "<script>" not in r.text                                        # no inline executable scripts
