"""Offline, resumable Jikan v4 enrichment (read-only; MyAnimeList data via the unofficial Jikan API).

Fetches /anime/{id}/full (themes, demographics, relations, synonyms),
/anime/{id}/recommendations (weak similarity labels) and optionally
/anime/{id}/characters, and caches every response under
data/enriched/jikan/anime/<mal_id>.json with endpoint, fetched_at, HTTP
status and the SHA-256 of the body. Requests are spaced by
jikan.min_interval_s (default 1.1 s, below the documented 3 req/s and
60 req/min limits); 429 / 5xx / timeouts back off exponentially and honour
Retry-After. Serving never calls Jikan. Current Jikan scores are stored only
in this layer and never replace the 2023 snapshot values.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from app.core.config import Settings
from pipelines.common import latest_dataset_version, utc_now_iso, write_json
from pipelines.processed_io import ProcessedDataset

ENDPOINTS = {"full": "/anime/{id}/full", "recommendations": "/anime/{id}/recommendations",
             "characters": "/anime/{id}/characters"}


class JikanClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        cfg = settings.jikan
        self.base = cfg.get("base_url", "https://api.jikan.moe/v4").rstrip("/")
        self.min_interval = float(cfg.get("min_interval_s", 1.1))
        self.max_retries = int(cfg.get("max_retries", 4))
        # Jikan answers 504 while it fetches an uncached entry upstream and serves it from cache on a later
        # request, so 5xx are not retried inline by default: the next resumable pass collects them.
        self.server_error_retries = int(cfg.get("server_error_retries", 0))
        self.client = httpx.Client(timeout=float(cfg.get("timeout_s", 20)), transport=transport,
                                   headers={"User-Agent": "AniLogic/2.0 (NUS-ISS course project; offline enrichment)"})
        self.sleep = sleep
        self._last = 0.0
        self.stats = {"requests": 0, "retries": 0, "rate_limited": 0, "errors": 0}

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            self.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def get(self, path: str) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait()
            self.stats["requests"] += 1
            try:
                r = self.client.get(self.base + path)
            except httpx.TimeoutException:
                status, body, retry_after = "timeout", None, None
            except httpx.HTTPError as exc:
                status, body, retry_after = f"error:{type(exc).__name__}", None, None
            else:
                status, body = r.status_code, r.content
                retry_after = r.headers.get("retry-after")
                if r.status_code == 200:
                    return {"endpoint": path, "status": 200, "fetched_at": utc_now_iso(),
                            "sha256": hashlib.sha256(body).hexdigest(), "data": r.json()}
                if r.status_code == 404:
                    return {"endpoint": path, "status": 404, "fetched_at": utc_now_iso(), "sha256": None, "data": None}
                if r.status_code == 429:
                    self.stats["rate_limited"] += 1
            # 429: honour Retry-After up to max_retries; 5xx / timeouts: one short retry, then leave it
            # for a later resumable pass instead of stalling the whole run on a failing upstream
            limit = self.max_retries if status == 429 else self.server_error_retries
            if attempt >= limit:
                break
            self.stats["retries"] += 1
            wait = float(retry_after) if retry_after and str(retry_after).isdigit() else (delay if status == 429 else 3.0)
            self.sleep(min(60.0, wait))
            delay *= 2
        self.stats["errors"] += 1
        return {"endpoint": path, "status": str(status), "fetched_at": utc_now_iso(), "sha256": None, "data": None}


def select_ids(settings: Settings, top_popular: int | None, ids: list[int] | None) -> list[int]:
    if ids:
        return sorted(set(int(i) for i in ids))
    ds = ProcessedDataset.for_version(settings, latest_dataset_version(settings))
    cat = ds.catalog()
    cat = cat[cat.in_catalog & cat.is_recommendable & cat.mal_id.notna()]
    cat = cat.sort_values(["members", "mal_id"], ascending=[False, True])
    return [int(x) for x in cat.mal_id.head(top_popular or 300)]


def enrich(settings: Settings, top_popular: int | None = None, ids: list[int] | None = None,
           with_characters: bool = True, force: bool = False, client: JikanClient | None = None, log=print,
           passes: int = 4, pause_between_passes: float = 60.0) -> dict:
    root = settings._abs(settings.real.get("enriched_dir", "data/enriched")) / "jikan" / "anime"
    root.mkdir(parents=True, exist_ok=True)
    client = client or JikanClient(settings)
    wanted = ["full", "recommendations"] + (["characters"] if with_characters else [])
    targets = select_ids(settings, top_popular, ids)
    t0 = time.time()
    for pass_no in range(1, passes + 1):
        done, skipped, failed = _pass(root, targets, wanted, force and pass_no == 1, client, t0, log)
        log(f"[jikan] pass {pass_no}: ok={done} cached={skipped} failed={failed}")
        if failed == 0:
            break
        if pass_no < passes:
            client.sleep(pause_between_passes)
    report = {"targets": len(targets), "fetched_ok": done, "skipped_cached": skipped, "failed": failed,
              "passes": pass_no, "client": client.stats, "min_interval_s": client.min_interval,
              "finished_at": utc_now_iso(), "cache_dir": str(root)}
    write_json(root.parent / "enrichment_report.json", report)
    log(f"[jikan] done: {report}")
    return report


def _pass(root: Path, targets: list[int], wanted: list[str], force: bool, client: JikanClient, t0: float, log
          ) -> tuple[int, int, int]:
    done = skipped = failed = 0
    for n, mid in enumerate(targets, start=1):
        path = root / f"{mid}.json"
        rec = json.loads(path.read_text()) if path.exists() else {"mal_id": mid, "endpoints": {}}
        missing = [e for e in wanted if force or (rec["endpoints"].get(e) or {}).get("status") != 200]
        if not missing:
            skipped += 1
            continue
        for e in missing:
            rec["endpoints"][e] = client.get(ENDPOINTS[e].format(id=mid))
        rec["fetched_at"] = utc_now_iso()
        write_json(path, rec)
        ok = all((rec["endpoints"].get(e) or {}).get("status") == 200 for e in wanted)
        done += int(ok)
        failed += int(not ok)
        if n % 25 == 0:
            log(f"[jikan] {n}/{len(targets)} ok={done} skipped={skipped} failed={failed} "
                f"elapsed={time.time() - t0:.0f}s stats={client.stats}")
    return done, skipped, failed


def cache_path(settings: Settings) -> Path:
    return settings._abs(settings.real.get("enriched_dir", "data/enriched")) / "jikan" / "anime"
