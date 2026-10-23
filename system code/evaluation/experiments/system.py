"""System-level evaluation: latency (cold / warm cache, concurrency), failure injection, robustness.

Runs the real FastAPI app in-process (TestClient) against a throwaway copy of the activated database.
Latencies are measured on this machine (reported in manifest.json -> hardware); they are observations,
not the targets from the design document.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from evaluation.common import Run, md_table
from evaluation.sandbox import sandbox_settings


def _pct(vals):
    return {"p50": round(float(np.percentile(vals, 50)), 1), "p95": round(float(np.percentile(vals, 95)), 1),
            "n": len(vals)} if vals else {"p50": None, "p95": None, "n": 0}


def run(settings, tag=None, requests_per_endpoint: int = 30, **_) -> dict:
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services import recommend

    run = Run(settings, "system", {"requests_per_endpoint": requests_per_endpoint, "tag": tag})
    rng = np.random.default_rng(71)
    metrics: dict = {}
    with sandbox_settings(settings) as sb:
        app = create_app(sb)
        with TestClient(app) as c:
            r = c.post("/api/v1/auth/register", json={"username": "eval_system_user", "password": "evaluation pw 1"})
            h = {"X-CSRF-Token": r.json()["csrf_token"]}
            st = app.state.holder.get()
            snap = st.snap
            pool = np.flatnonzero(snap.recommendable & snap.in_catalog)
            popular = pool[np.argsort(-np.nan_to_num(snap.members[pool]))][:500]
            for i in rng.choice(popular[:200], size=10, replace=False):
                c.put(f"/api/v1/anime/{snap.keys[i]}/rating", json={"rating": int(rng.integers(6, 11))}, headers=h)
            cid = c.post("/api/v1/collections", json={"name": "eval"}, headers=h).json()["id"]
            for i in popular[:5]:
                c.put(f"/api/v1/collections/{cid}/items/{snap.keys[i]}", headers=h)
            sid = c.post("/api/v1/chat/sessions", json={}, headers=h).json()["id"]
            words = [w for i in popular[:300] for w in snap.titles[i].split() if len(w) > 3][:300]
            queries = ["a comedy with at most 12 episodes", "something short and relaxing, no horror",
                       "a mystery with a detective and at most 24 episodes", "an action anime with robots",
                       "romance at school, preferably short"]
            timings: dict[str, list[float]] = {}

            def timed(name, fn, clear_cache=False):
                if clear_cache:
                    recommend.CACHE.clear()
                t0 = time.perf_counter()
                resp = fn()
                timings.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
                if resp.status_code >= 500:
                    run.failure({"endpoint": name, "status": resp.status_code})
                return resp

            for k in range(requests_per_endpoint):
                key = str(snap.keys[rng.choice(popular)])
                timed("search", lambda: c.get("/api/v1/anime", params={"q": rng.choice(words)}))
                timed("detail", lambda key=key: c.get(f"/api/v1/anime/{key}"))
                timed("similar_cold", lambda key=key: c.get(f"/api/v1/anime/{key}/similar"), clear_cache=True)
                timed("similar_warm", lambda key=key: c.get(f"/api/v1/anime/{key}/similar"))
                timed("home_cold", lambda: c.post("/api/v1/recommendations/home", json={}, headers=h), clear_cache=True)
                timed("home_warm", lambda: c.post("/api/v1/recommendations/home", json={}, headers=h))
                timed("collection_recs", lambda: c.post(f"/api/v1/collections/{cid}/recommendations", json={}, headers=h))
                timed("chat_local", lambda k=k: c.post(f"/api/v1/chat/sessions/{sid}/messages",
                                                       json={"message": queries[k % len(queries)]}, headers=h))
                timed("plan_preview", lambda: c.post("/api/v1/plans/preview",
                                                     json={"budget_minutes": int(rng.choice([300, 900, 1500])),
                                                           "max_titles": 3}, headers=h))
            metrics["latency_ms"] = {k: _pct(v) for k, v in timings.items()}

            # ---- concurrency 5 (mixed reads)
            conc = []

            def worker(n):
                out = []
                for j in range(12):
                    key = str(snap.keys[popular[(n * 13 + j) % len(popular)]])
                    t0 = time.perf_counter()
                    resp = c.get(f"/api/v1/anime/{key}/similar") if j % 2 else c.get("/api/v1/anime", params={"q": words[j]})
                    out.append(((time.perf_counter() - t0) * 1000, resp.status_code))
                return out

            with ThreadPoolExecutor(max_workers=5) as ex:
                for part in ex.map(worker, range(5)):
                    conc.extend(part)
            metrics["concurrency5_mixed_reads"] = {**_pct([x for x, _ in conc]),
                                                   "errors": sum(1 for _, s in conc if s >= 500)}

            # ---- failure injection
            from app.chat import llm as llm_mod
            from app.services import chat as chat_svc

            fails = {}
            for err in ("timeout", "rate_limited", "invalid_json", "empty", "refusal"):
                calls = []

                class Stub:
                    def complete(self, *a, _e=err, _calls=calls, **k):
                        _calls.append(_e)
                        return llm_mod.LLMResult(ok=False, provider="stub", model="m", error=_e)
                orig = chat_svc.get_llm
                chat_svc.get_llm = lambda s, _S=Stub: _S()
                try:
                    # a fresh session per case, so earlier turns cannot narrow the candidates to nothing
                    fresh = c.post("/api/v1/chat/sessions", json={}, headers=h).json()["id"]
                    resp = c.post(f"/api/v1/chat/sessions/{fresh}/messages", json={"message": queries[0]}, headers=h)
                    body = resp.json()
                    fell_back = any("language model was unavailable" in w for w in body.get("warnings", []))
                    fails[f"llm_{err}"] = {"status": resp.status_code, "answer_mode": body.get("answer_mode"),
                                           "items": len(body.get("items", [])), "llm_calls": len(calls),
                                           "fallback_warning": fell_back,
                                           "ok": resp.status_code == 200 and len(calls) >= 1 and fell_back
                                           and body.get("answer_mode") == "local_template" and bool(body.get("items"))}
                finally:
                    chat_svc.get_llm = orig
            dense = st.dense
            if dense is not None:
                def broken(*a, **k):
                    raise RuntimeError("encoder unavailable")
                orig_search = dense.search
                dense.search = broken
                try:
                    resp = c.post(f"/api/v1/chat/sessions/{sid}/messages",
                                  json={"message": "a story about a lonely robot who learns to paint"}, headers=h)
                    body = resp.json()
                    fails["dense_encoder_failure"] = {"status": resp.status_code, "items": len(body.get("items", [])),
                                                      "warning_shown": any("lexical" in w for w in body.get("warnings", [])),
                                                      "ok": resp.status_code == 200}
                finally:
                    dense.search = orig_search
            long_msg = ("please find me an anime " * 60)[:1000]
            resp = c.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": long_msg}, headers=h)
            fails["long_input_1000_chars"] = {"status": resp.status_code, "ok": resp.status_code == 200}
            resp = c.post(f"/api/v1/chat/sessions/{sid}/messages", json={"message": "x" * 1001}, headers=h)
            fails["over_limit_input_rejected"] = {"status": resp.status_code, "ok": resp.status_code == 422}
            resp = c.post(f"/api/v1/chat/sessions/{sid}/messages",
                          json={"message": "<script>alert(1)</script> a comedy"}, headers=h)
            fails["html_injection_is_data"] = {"status": resp.status_code, "ok": resp.status_code == 200}
            resp = c.get("/api/v1/anime", params={"max_episodes": 1, "min_score": 9.99})
            fails["extreme_filters"] = {"status": resp.status_code, "total": resp.json().get("total"),
                                        "ok": resp.status_code == 200}
            resp = c.post(f"/api/v1/chat/sessions/{sid}/messages",
                          json={"message": "at least 50 episodes but at most 3 episodes"}, headers=h)
            fails["contradiction_clarified"] = {"ok": bool(resp.json().get("clarification"))}

            # ---- concurrent writes (SQLite contention): 6 users rating in parallel, one transaction per write
            from app.db.models import User
            from app.db.session import session_scope, write_retry
            from app.services import auth as auth_svc
            from app.services import user_data

            uids = []
            for n in range(6):
                with session_scope(sb.db_url) as db:
                    uids.append(auth_svc.register(db, f"eval_writer_{n}", "evaluation pw 1", 1).user.id)

            def writer(n):
                errs = 0
                for j in range(15):
                    key = str(snap.keys[popular[(n * 17 + j) % len(popular)]])

                    def once(key=key):
                        with session_scope(sb.db_url) as db:
                            user_data.put_rating(db, db.get(User, uids[n]), key, 7)
                    try:
                        write_retry(once)
                    except Exception:  # noqa: BLE001 - counted as a failed write
                        errs += 1
                return errs
            with ThreadPoolExecutor(max_workers=6) as ex:
                write_errors = sum(ex.map(writer, range(6)))
            fails["concurrent_rating_writes"] = {"writes": 90, "failed_writes": write_errors, "ok": write_errors == 0}
            metrics["failure_injection"] = fails
    metrics["llm_cost_per_chat"] = "not applicable: no LLM configured (local mode); cost is reported as unknown"
    lat_rows = [{"endpoint": k, **v} for k, v in metrics["latency_ms"].items()]
    run.comparison(lat_rows)
    lines = [f"# System evaluation - {settings.mode} data", "",
             "In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). "
             "`*_cold` clears the recommendation cache before each call.", "",
             md_table(lat_rows, ["endpoint", "p50", "p95", "n"]), "",
             f"Concurrency 5 (mixed reads): {metrics['concurrency5_mixed_reads']}", "",
             "## Failure injection and robustness", "",
             md_table([{"case": k, **v} for k, v in metrics["failure_injection"].items()], ["case", "ok", "status"]), "",
             "LLM token cost is unknown / not applicable in local mode; nothing is reported as 0."]
    return run.finish(metrics, "\n".join(lines) + "\n", {"catalog_titles": int(len(pool))})
