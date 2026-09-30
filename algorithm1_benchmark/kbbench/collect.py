"""Read-only public Jikan requests. Cached response bytes are evidence, never instructions."""
import argparse
import gzip
import json
import time
import tempfile
from pathlib import Path

from .common import ROOT, read_rows, sha256, utc_now, write_json, write_rows
from .network import download_file
from .mal_html import parse_recommendations


def fetch(url, attempts=1):
    for attempt in range(attempts):
        try:
            with tempfile.TemporaryDirectory(prefix="kbbench-mal-") as folder:
                path = Path(folder) / "response.json"
                download_file(url, path, timeout=30)
                body = path.read_bytes()
            payload = json.loads(body)
            if not isinstance(payload.get("data"), list):
                raise ValueError("Unexpected API schema")
            return body
        except (OSError, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(str(error)) from error
            time.sleep(2 ** (attempt + 1))


def fetch_html(aid, cache):
    url = f"https://myanimelist.net/anime/{aid}/_/userrecs"
    path = cache / f"{aid}.html.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            page = stream.read()
    else:
        with tempfile.TemporaryDirectory(prefix="kbbench-mal-page-") as folder:
            temp = Path(folder) / "page.html"
            download_file(url, temp, timeout=30)
            body = temp.read_bytes()
        page = body.decode("utf-8")
        # Validate before caching; blocked/challenge pages are not empty truth.
        parse_recommendations(page, aid)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(body)
        write_json(cache / f"{aid}.html.meta.json", dict(url=url, retrieved_at=utc_now(), sha256=sha256(path)))
        time.sleep(1.05)
    return json.dumps(parse_recommendations(page, aid), ensure_ascii=False).encode("utf-8"), url, path


def collect():
    ids = {row["anime_id"] for row in read_rows(ROOT / "data/catalog.jsonl.gz")}
    targets = list(read_rows(ROOT / "data/targets.jsonl"))
    cache = ROOT / "data/mal_cache"
    cache.mkdir(parents=True, exist_ok=True)
    labels, audit = [], []
    for number, target in enumerate(targets, 1):
        aid = target["anime_id"]
        path = cache / f"{aid}.json.gz"
        url = f"https://api.jikan.moe/v4/anime/{aid}/recommendations"
        try:
            if not path.exists():
                api_error = None
                raw_page = None
                try:
                    body = fetch(url)
                except RuntimeError as error:
                    api_error = str(error)
                    body, url, raw_page = fetch_html(aid, cache)
                # Keep full API response for vote/filter auditing, with deterministic gzip headers.
                with path.open("wb") as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                        stream.write(body)
                write_json(cache / f"{aid}.meta.json", dict(url=url, retrieved_at=utc_now(), sha256=sha256(path),
                           retrieval="MAL HTML fallback" if raw_page else "Jikan API",
                           raw_page=raw_page.name if raw_page else None, jikan_error=api_error))
                time.sleep(1.05)
            metadata = json.loads((cache / f"{aid}.meta.json").read_text(encoding="utf-8"))
            url = metadata["url"]
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                response = json.load(stream)
            recommendations = {}
            for rec in response["data"]:
                rid, votes = int(rec["entry"]["mal_id"]), int(rec.get("votes", 0))
                if rid in ids and rid != aid and votes > 0:
                    recommendations[rid] = max(votes, recommendations.get(rid, 0))
            ranked = sorted(recommendations, key=lambda rid: (-recommendations[rid], rid))[:5]
            eligible = len(ranked) >= 3
            audit.append(dict(**target, status="eligible" if eligible else "insufficient_labels", raw_count=len(response["data"]),
                              eligible_recommendations=len(recommendations), selected_count=len(ranked), cache_sha256=sha256(path)))
            if eligible:
                labels.append(dict(target_id=aid, relevant_ids=ranked, votes=[recommendations[rid] for rid in ranked],
                                   source="mal_recommendations", source_url=url, split=target["split"],
                                   sampling_group=target["sampling_group"], cache_file=str(path.relative_to(ROOT)).replace("\\", "/")))
            print(f"{number}/{len(targets)} MAL {aid}: {len(recommendations)} catalog matches", flush=True)
        except (RuntimeError, OSError, ValueError) as error:
            audit.append(dict(**target, status="request_failed", error=str(error)))
            print(f"{number}/{len(targets)} MAL {aid}: FAILED (see collection_audit.json)", flush=True)
        write_rows(ROOT / "data/mal_ground_truth.jsonl", labels)
        write_json(ROOT / "data/collection_audit.json", audit)
    print(f"Collected {len(labels)} eligible queries from {len(targets)} predeclared targets", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    collect()
