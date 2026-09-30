"""Real data: validate and preprocess the Kaggle MAL 2023 snapshot (+ optional Jikan layer).

The raw files are never modified. Every exclusion is counted with a reason.
Offline users keep the Kaggle ``user_id`` namespace; usernames, gender,
birthday and location are never imported.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.ids import mal_key
from pipelines import cleaning as C
from pipelines.common import (
    PIPELINE_VERSION,
    dataset_version,
    dependency_versions,
    peak_rss_mb,
    read_json,
    rel,
    sha256_file,
    utc_now_iso,
    write_json,
)
from pipelines.processed_io import ProcessedWriter, mix_hash

ANIME_REQUIRED = ["anime_id", "Name", "Score", "Genres", "Type", "Episodes", "Duration", "Rating", "Status"]
ANIME_OPTIONAL = ["English name", "Other name", "Synopsis", "Aired", "Premiered", "Producers", "Licensors",
                  "Studios", "Source", "Rank", "Popularity", "Favorites", "Scored By", "Members", "Image URL"]
SCORES_REQUIRED = ["user_id", "anime_id", "rating"]
DETAILS_EXPECTED = ["Mal ID", "Username"]


class DataValidationError(RuntimeError):
    pass


def _raw_dir(settings: Settings) -> Path:
    return settings._abs(settings.real.get("raw_dir", "data/raw/mal_2023"))


def _scores_path(settings: Settings) -> Path | None:
    raw = _raw_dir(settings)
    for name in (settings.real.get("scores_file", "users-score-2023.csv"),
                 settings.real.get("scores_file_gz", "users-score-2023.csv.gz")):
        p = raw / name
        if p.exists():
            return p
    return None


def _count_rows(path: Path) -> int:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    n = -1  # header
    with opener(path, "rb") as fh:
        for _ in fh:
            n += 1
    return max(n, 0)


def validate_real(settings: Settings, count_rows: bool = True) -> dict[str, Any]:
    raw = _raw_dir(settings)
    report: dict[str, Any] = {"checked_at": utc_now_iso(), "raw_dir": rel(raw), "files": {}, "errors": [],
                              "warnings": []}
    anime_path = raw / settings.real.get("anime_file", "anime-dataset-2023.csv")
    scores_path = _scores_path(settings)
    details_path = raw / settings.real.get("details_file", "users-details-2023.csv")

    def check(path: Path | None, label: str, required: list[str], optional: list[str] | None = None,
              needed: bool = True) -> None:
        if path is None or not path.exists():
            (report["errors"] if needed else report["warnings"]).append(f"{label}: file not found under {rel(raw)}")
            return
        head = pd.read_csv(path, nrows=200, dtype=str, keep_default_na=False)
        missing = [c for c in required if c not in head.columns]
        info = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                "columns": list(head.columns), "missing_required_columns": missing}
        if optional:
            info["missing_optional_columns"] = [c for c in optional if c not in head.columns]
        if count_rows:
            info["rows"] = _count_rows(path)
        report["files"][label] = info
        if missing:
            report["errors"].append(f"{label}: missing required columns {missing}")

    check(anime_path, "anime", ANIME_REQUIRED, ANIME_OPTIONAL)
    check(scores_path, "scores", SCORES_REQUIRED)
    check(details_path, "details", DETAILS_EXPECTED, needed=False)
    if (raw / "final_animedataset.csv").exists():
        report["warnings"].append("final_animedataset.csv present: ignored by design (duplicated wide table)")
    report["ok"] = not report["errors"]
    write_json(settings.processed_dir / "validation_report.json", report)
    return report


# ---------------------------------------------------------------------- enrichment


def load_jikan_layer(settings: Settings) -> dict[int, dict]:
    """Read cached Jikan responses (data/enriched/jikan/anime/<mal_id>.json)."""
    root = settings._abs(settings.real.get("enriched_dir", "data/enriched")) / "jikan" / "anime"
    out: dict[int, dict] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*.json")):
        try:
            rec = read_json(p)
            out[int(rec["mal_id"])] = rec
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return out


def _jikan_data(rec: dict, endpoint: str) -> Any:
    ep = (rec.get("endpoints") or {}).get(endpoint) or {}
    return (ep.get("data") or {}).get("data")


# ---------------------------------------------------------------------- preprocess


def preprocess_real(settings: Settings, log=print) -> dict[str, Any]:
    t0 = time.time()
    validation = validate_real(settings, count_rows=False)
    if not validation["ok"]:
        raise DataValidationError("; ".join(validation["errors"]))
    raw = _raw_dir(settings)
    anime_path = raw / settings.real.get("anime_file", "anime-dataset-2023.csv")
    scores_path = _scores_path(settings)
    assert scores_path is not None
    source_hashes = {"anime": validation["files"]["anime"]["sha256"],
                     "scores": validation["files"]["scores"]["sha256"]}
    jikan = load_jikan_layer(settings)
    if jikan:
        source_hashes["jikan_layer"] = hashlib_of_layer(jikan)
    pipe = settings.pipeline
    config_fp = settings.fingerprint("content_rules") + ":" + json.dumps(
        {k: pipe.get(k) for k in ("sample_users_fraction", "sample_seed")}, sort_keys=True)
    version = dataset_version("real", source_hashes, config_fp)
    root = settings.processed_dir / version
    writer = ProcessedWriter(root, int(pipe.get("rating_partitions", 32)))
    log(f"[preprocess-real] dataset_version={version}")

    # ---------------------------------------------------------------- anime
    df = pd.read_csv(anime_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rules = settings.content_rules
    catalog_rows, tag_rows, alias_rows, relation_rows, char_rows = [], [], [], [], []
    exclusion = Counter()
    quarantine = []
    seen_ids: set[int] = set()
    for rec in df.to_dict("records"):
        mal_id = C.parse_int(rec.get("anime_id"))
        title = C.clean_str(rec.get("Name"))
        if mal_id is None or mal_id == 0:
            quarantine.append({"anime_id": rec.get("anime_id"), "reason": "invalid_id"})
            continue
        if mal_id in seen_ids:
            quarantine.append({"anime_id": mal_id, "reason": "duplicate_id"})
            continue
        seen_ids.add(mal_id)
        if title is None:
            quarantine.append({"anime_id": mal_id, "reason": "missing_title"})
            continue
        key = mal_key(mal_id)
        genres = C.parse_multi(rec.get("Genres"))
        studios = C.parse_multi(rec.get("Studios"))
        producers = C.parse_multi(rec.get("Producers"))
        licensors = C.parse_multi(rec.get("Licensors"))
        source_material = C.clean_str(rec.get("Source"))
        episodes = C.parse_int(rec.get("Episodes"))
        if episodes == 0:
            episodes = None
        dur = C.parse_duration(rec.get("Duration"))
        per_ep, total, basis = C.resolve_runtime(dur, episodes)
        aired_from, aired_to = C.parse_aired(rec.get("Aired"))
        season, prem_year = C.parse_premiered(rec.get("Premiered"))
        score = C.parse_float(rec.get("Score"), 1.0, 10.0)
        content_rating = C.clean_str(rec.get("Rating"))
        explicit = C.is_explicit(genres, content_rating, rules.exclude_genres, rules.exclude_rating_prefixes)
        synopsis = C.parse_synopsis(rec.get("Synopsis"))
        typ = C.clean_str(rec.get("Type"))
        status = C.clean_str(rec.get("Status"))

        themes, demos, synonyms = [], [], []
        jrec = jikan.get(mal_id)
        if jrec:
            full = _jikan_data(jrec, "full") or {}
            themes = [t["name"] for t in full.get("themes") or [] if t.get("name")]
            demos = [t["name"] for t in full.get("demographics") or [] if t.get("name")]
            synonyms = [s for s in full.get("title_synonyms") or [] if s]
            for rel_group in full.get("relations") or []:
                rtype = str(rel_group.get("relation", "")).strip().lower().replace(" ", "_")
                for entry in rel_group.get("entry") or []:
                    relation_rows.append({
                        "from_mal": mal_id, "to_mal": entry.get("mal_id"), "to_type": entry.get("type"),
                        "to_name": entry.get("name"), "relation_type": rtype,
                        "source_ref": f"jikan:/anime/{mal_id}/full#relations"})
            chars = _jikan_data(jrec, "characters") or []
            for c in chars:
                ch = c.get("character") or {}
                if ch.get("mal_id") and ch.get("name"):
                    char_rows.append({"source_id": f"mal-character:{ch['mal_id']}", "name": ch["name"],
                                      "item_key": key, "role": c.get("role"),
                                      "source_ref": f"jikan:/anime/{mal_id}/characters"})

        missing = [f for f, v in (("score", score), ("episodes", episodes), ("duration", per_ep),
                                  ("studios", studios), ("genres", genres), ("synopsis", synopsis),
                                  ("aired", aired_from), ("type", typ)) if v in (None, [])]
        reasons = []
        if explicit:
            reasons.append("content_rule_explicit")
        in_catalog = not explicit
        recommendable = in_catalog and score is not None and bool(genres)
        if in_catalog and score is None:
            reasons.append("not_recommendable_missing_score")
        if in_catalog and not genres:
            reasons.append("not_recommendable_missing_genre")
        planner_ok = (recommendable and status == "Finished Airing" and episodes is not None
                      and per_ep is not None and typ is not None)
        for r in reasons:
            exclusion[r] += 1

        catalog_rows.append({
            "item_key": key, "mal_id": mal_id, "title": title,
            "title_english": C.clean_str(rec.get("English name")),
            "title_other": C.clean_str(rec.get("Other name")), "synopsis": synopsis, "score": score,
            "scored_by": C.parse_int(rec.get("Scored By")), "rank": C.parse_int(rec.get("Rank")),
            "popularity": C.parse_int(rec.get("Popularity")), "members": C.parse_int(rec.get("Members")),
            "favorites": C.parse_int(rec.get("Favorites")), "type": typ, "episodes": episodes,
            "duration_minutes": per_ep, "total_minutes": total, "duration_raw": dur.raw,
            "duration_basis": basis, "status": status, "content_rating": content_rating,
            "source_material": source_material, "aired_raw": C.clean_str(rec.get("Aired")),
            "aired_from": aired_from, "aired_to": aired_to, "premiered_season": season,
            "premiered_year": prem_year, "cover_url": C.clean_str(rec.get("Image URL")),
            "provenance": {"source": "kaggle:dbdmobile/myanimelist-dataset:anime-dataset-2023.csv",
                           "snapshot": "2023", "jikan_enriched": bool(jrec),
                           "jikan_fetched_at": (jrec or {}).get("fetched_at")},
            "missing_fields": missing, "in_catalog": in_catalog, "is_recommendable": recommendable,
            "is_planner_eligible": planner_ok, "is_explicit": explicit, "exclusion_reasons": reasons,
        })
        for ns, values, src in (("genre", genres, "kaggle"), ("studio", studios, "kaggle"),
                                ("producer", producers, "kaggle"), ("licensor", licensors, "kaggle"),
                                ("theme", themes, "jikan"), ("demographic", demos, "jikan")):
            for v in values:
                tag_rows.append({"item_key": key, "namespace": ns, "value": v, "source": src})
        if source_material:
            tag_rows.append({"item_key": key, "namespace": "source", "value": source_material, "source": "kaggle"})
        alias_rows.append({"item_key": key, "alias": title, "kind": "title"})
        for kind, val in (("english", C.clean_str(rec.get("English name"))),
                          ("other", C.clean_str(rec.get("Other name")))):
            if val and val != title:
                alias_rows.append({"item_key": key, "alias": val, "kind": kind})
        for s in synonyms:
            alias_rows.append({"item_key": key, "alias": s, "kind": "synonym"})

    catalog = pd.DataFrame(catalog_rows).sort_values("mal_id").reset_index(drop=True)
    writer.write_catalog(catalog)
    writer.write_table("item_tags", pd.DataFrame(tag_rows, columns=["item_key", "namespace", "value", "source"]))
    writer.write_table("aliases", pd.DataFrame(alias_rows, columns=["item_key", "alias", "kind"]).drop_duplicates())
    valid_keys = set(catalog.loc[catalog.in_catalog, "item_key"])
    rel_df = _resolve_relations(relation_rows, valid_keys)
    writer.write_table("relations", rel_df)
    writer.write_table("characters", pd.DataFrame(
        [r for r in char_rows if r["item_key"] in valid_keys],
        columns=["source_id", "name", "item_key", "role", "source_ref"]))
    labels = _weak_similarity_labels(jikan, valid_keys)
    with (root / "labels" / "similarity_mal_recommendations.jsonl").open("w", encoding="utf-8") as fh:
        for row in labels:
            fh.write(json.dumps(row) + "\n")
    log(f"[preprocess-real] catalogue parsed: {len(catalog)} titles, in_catalog={int(catalog.in_catalog.sum())}, "
        f"recommendable={int(catalog.is_recommendable.sum())}, planner={int(catalog.is_planner_eligible.sum())}")

    # ---------------------------------------------------------------- ratings (chunked)
    mal_to_idx = np.full(int(catalog.mal_id.max()) + 1, -1, dtype=np.int64)
    mal_to_idx[catalog.mal_id.to_numpy()] = np.arange(len(catalog))
    served_mask = catalog.in_catalog.to_numpy()
    frac = pipe.get("sample_users_fraction")
    sample_seed = int(pipe.get("sample_seed", 7))
    chunk_rows = int(pipe.get("chunk_rows", 100_000))
    counts = Counter()
    users_before: Counter = Counter()
    users_after: Counter = Counter()
    rating_hist_before = np.zeros(11, dtype=np.int64)
    rating_hist_after = np.zeros(11, dtype=np.int64)
    n_chunks = 0
    reader = pd.read_csv(scores_path, usecols=SCORES_REQUIRED, chunksize=chunk_rows,
                         dtype={"user_id": "int64", "anime_id": "int64", "rating": "float64"})
    for chunk in reader:
        n_chunks += 1
        counts["rows_read"] += len(chunk)
        r = chunk["rating"].to_numpy()
        valid_r = np.isfinite(r) & (r == np.round(r)) & (r >= 1) & (r <= 10)
        counts["invalid_rating"] += int((~valid_r).sum())
        a = chunk["anime_id"].to_numpy()
        u = chunk["user_id"].to_numpy()
        in_range = (a >= 0) & (a < len(mal_to_idx))
        idx = np.where(in_range, mal_to_idx[np.clip(a, 0, len(mal_to_idx) - 1)], -1)
        unknown = valid_r & (idx < 0)
        counts["quarantine_unknown_anime_id"] += int(unknown.sum())
        known = valid_r & (idx >= 0)
        excluded = known & ~served_mask[np.clip(idx, 0, len(served_mask) - 1)]
        counts["dropped_content_rule_item"] += int(excluded.sum())
        keep = known & ~excluded
        np.add.at(rating_hist_before, r[keep].astype(np.int64), 1)
        uu, cc = np.unique(u[keep], return_counts=True)
        users_before.update(dict(zip(uu.tolist(), cc.tolist(), strict=True)))
        if frac is not None:
            keep &= mix_hash(u, sample_seed) < float(frac)
        np.add.at(rating_hist_after, r[keep].astype(np.int64), 1)
        uu, cc = np.unique(u[keep], return_counts=True)
        users_after.update(dict(zip(uu.tolist(), cc.tolist(), strict=True)))
        writer.append_ratings(u[keep], idx[keep], r[keep].astype(np.int8))
        counts["rows_kept"] += int(keep.sum())
        if n_chunks % 40 == 0:
            log(f"[preprocess-real] chunks={n_chunks} rows_read={counts['rows_read']:,} rss={peak_rss_mb()}MB")
    dedupe = writer.close_ratings()

    def _dist(cnt: Counter) -> dict:
        if not cnt:
            return {"users": 0}
        arr = np.fromiter(cnt.values(), dtype=np.int64)
        return {"users": int(len(arr)), "ratings": int(arr.sum()),
                "ratings_per_user_pct": {str(q): float(np.percentile(arr, q)) for q in (5, 25, 50, 75, 95, 99)}}

    report = {
        "dataset_version": version, "generated_at": utc_now_iso(), "elapsed_s": round(time.time() - t0, 1),
        "peak_rss_mb": peak_rss_mb(), "chunks": n_chunks, "chunk_rows": chunk_rows,
        "catalog": {
            "rows_in_snapshot": int(len(df)), "parsed": int(len(catalog)),
            "quarantined": len(quarantine), "in_catalog": int(catalog.in_catalog.sum()),
            "explicit_excluded": int(catalog.is_explicit.sum()),
            "recommendable": int(catalog.is_recommendable.sum()),
            "planner_eligible": int(catalog.is_planner_eligible.sum()),
            "reasons": dict(exclusion),
            "missing_rate": {f: round(float(catalog.missing_fields.map(lambda m, f=f: f in m).mean()), 4)
                             for f in ("score", "studios", "genres", "episodes", "duration", "synopsis")},
            "jikan_enriched_titles": int(sum(1 for p in catalog.provenance if p.get("jikan_enriched"))),
        },
        "ratings": {**{k: int(v) for k, v in counts.items()}, **dedupe,
                    "sampling": {"sample_users_fraction": frac, "sample_seed": sample_seed,
                                 "hash": "splitmix64(user_id ^ seed) < fraction",
                                 "before": {**_dist(users_before), "rating_hist": rating_hist_before.tolist()},
                                 "after": {**_dist(users_after), "rating_hist": rating_hist_after.tolist()}}},
        "memory_note": _memory_note(len(users_after), int(catalog.in_catalog.sum()), dedupe["rows"]),
        "relations": {"resolved_anime_edges": int((rel_df.to_entity_type == "anime").sum()) if len(rel_df) else 0,
                      "external_edges": int((rel_df.to_entity_type != "anime").sum()) if len(rel_df) else 0},
        "weak_similarity_label_pairs": len(labels),
    }
    writer.write_report("preprocess_report", report)
    with (root / "reports" / "quarantine.jsonl").open("w", encoding="utf-8") as fh:
        for q in quarantine:
            fh.write(json.dumps(q) + "\n")
    manifest = writer.finalize({
        "dataset_mode": "real", "dataset_version": version, "pipeline_version": PIPELINE_VERSION,
        "source_hashes": source_hashes, "config_fingerprint": config_fp, "created_at": utc_now_iso(),
        "counts": {"catalog": int(len(catalog)), "ratings": dedupe["rows"], "users": len(users_after)},
        "dependencies": dependency_versions(),
    })
    write_json(settings.processed_dir / "latest.json", {"dataset_version": version, "updated_at": utc_now_iso()})
    log(f"[preprocess-real] done in {report['elapsed_s']}s: {dedupe['rows']:,} ratings, {len(users_after):,} users")
    return manifest


def hashlib_of_layer(jikan: dict[int, dict]) -> str:
    import hashlib

    h = hashlib.sha256()
    for mid in sorted(jikan):
        for name, ep in sorted((jikan[mid].get("endpoints") or {}).items()):
            h.update(f"{mid}:{name}:{ep.get('sha256', '')}".encode())
    return h.hexdigest()


def _memory_note(n_users: int, n_items: int, nnz: int) -> dict:
    dense_bytes = n_users * n_items * 4
    return {"shape": [n_users, n_items], "dense_float32_gb": round(dense_bytes / 1e9, 2),
            "csr_float32_mb": round((nnz * (4 + 4) + (n_users + 1) * 8) / 1e6, 1),
            "note": "dense user x item arrays are never materialised; CSR/CSC is used"}


def _resolve_relations(rows: list[dict], valid_keys: set[str]) -> pd.DataFrame:
    out = []
    for r in rows:
        from_key = mal_key(r["from_mal"])
        if from_key not in valid_keys:
            continue
        if r.get("to_type") == "anime" and r.get("to_mal"):
            to_key = mal_key(r["to_mal"])
            if to_key not in valid_keys:
                out.append({**_rel_base(r, from_key), "to_item": None, "to_entity_type": "anime_unresolved",
                            "to_external_id": str(r["to_mal"])})
                continue
            out.append({**_rel_base(r, from_key), "to_item": to_key, "to_entity_type": "anime",
                        "to_external_id": None})
        else:
            out.append({**_rel_base(r, from_key), "to_item": None,
                        "to_entity_type": str(r.get("to_type") or "unknown"),
                        "to_external_id": str(r.get("to_mal")) if r.get("to_mal") else None})
    cols = ["from_item", "to_item", "to_entity_type", "to_external_id", "to_name", "relation_type", "status",
            "source_ref"]
    return pd.DataFrame(out, columns=cols).drop_duplicates(["from_item", "to_item", "to_external_id",
                                                           "relation_type"])


def _rel_base(r: dict, from_key: str) -> dict:
    return {"from_item": from_key, "to_name": r.get("to_name"), "relation_type": r["relation_type"],
            "status": "confirmed", "source_ref": r["source_ref"]}


def _weak_similarity_labels(jikan: dict[int, dict], valid_keys: set[str]) -> list[dict]:
    """MAL user recommendation pairs -> weak similarity labels (never human ground truth)."""
    rows = []
    for mid, rec in jikan.items():
        data = _jikan_data(rec, "recommendations") or []
        q = mal_key(mid)
        if q not in valid_keys:
            continue
        pos = []
        for d in data:
            entry = d.get("entry") or {}
            k = mal_key(entry["mal_id"]) if entry.get("mal_id") else None
            if k and k in valid_keys and k != q:
                pos.append({"item_key": k, "votes": int(d.get("votes") or 0)})
        if pos:
            pos.sort(key=lambda x: (-x["votes"], x["item_key"]))
            rows.append({"query_item": q, "positives": pos[:20], "label_source": "mal_user_recommendations",
                         "label_strength": "weak", "source_ref": f"jikan:/anime/{mid}/recommendations"})
    return rows
