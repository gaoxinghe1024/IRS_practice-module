"""Import the team's curated data package (AnimeFinder system-dev release) as supplementary inputs.

The package (see its SYSTEM_DEV_HANDOFF.md) starts from the same Kaggle MAL 2023 snapshot as this repository
and adds themes, demographics, characters, relations, a few synopses, 300 sourced scenes and 100 retrieval
evaluation queries. This module converts the parts the system uses into the repository's existing contracts
(docs/data-contracts.md); the normal real pipeline then ingests them into a new dataset version.

Read from the package (nothing else is opened; in particular data/canonical/users.csv.gz with usernames and
gender is never read):
    DATASET_INDEX.json                    catalogue version and status (must be "final")
    data/canonical/anime_all.jsonl        themes, demographics, characters, relations, synopses per MAL id
    data/rag/scenes_verified.jsonl        scene facts with their sources
    data/rag/evaluation_queries.jsonl     retrieval evaluation queries (written to data/evaluation/, never indexed)

Written:
    data/enriched/tags.jsonl              theme / demographic values
    data/enriched/synopses.jsonl          English synopses for titles whose 2023 synopsis is missing
    data/enriched/relations.jsonl         anime -> anime relations
    data/enriched/characters.jsonl        character lists
    data/enriched/scenes.jsonl            scenes (verification method and checker kept in source_ref)
    data/evaluation/rag_queries.jsonl     label_source = team_package_<version>
    data/enriched/team_package_import.json  report: versions, hashes, counts, policies

Policies
  * Genres, studios, episodes, durations and scores are NOT imported: the 2023 snapshot stays the single source
    of these core fields (no 2026 values mixed in), and the frozen algorithm-2 model is bound to them.
  * Demographics: one value per title, the alphabetically first when the package lists several. This is the
    rule of the team's algorithm-1 projection that the frozen algorithm-2 model was trained with, so both
    algorithms see one consistent catalogue (45 runtime titles lose a second demographic).
  * Only records with a confirmed MAL identity are used; values without a non-snapshot provenance entry are kept
    with source "team_package".
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import Settings
from pipelines.common import sha256_file, utc_now_iso, write_json

CATALOG_FILE = "data/canonical/anime_all.jsonl"
SCENES_FILE = "data/rag/scenes_verified.jsonl"
QUERIES_FILE = "data/rag/evaluation_queries.jsonl"
INDEX_FILE = "DATASET_INDEX.json"
NOT_READ = ["data/canonical/users.csv.gz (usernames, gender: never imported)",
            "data/canonical/ratings_runtime.csv.gz and data/cf/* (ratings come from the raw Kaggle files)",
            "data/knowledge/*, data/ga/*, data/rag/*_documents.jsonl (rebuilt by this system's pipelines)"]
SNAPSHOT_SOURCES = {"anime2023"}            # the Kaggle table this repository already reads


class PackageError(RuntimeError):
    pass


def _json_field(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) and value[:1] in "[{" else value


def _sources(prov: dict, field: str) -> list[str]:
    """Non-snapshot provenance sources of a field ("anilist_detail", "external_verified", ...)."""
    out = []
    for p in prov.get(field) or []:
        s = p.get("source") if isinstance(p, dict) else None
        if s and s not in SNAPSHOT_SOURCES and s not in out:
            out.append(s)
    return out


def _display_path(path: Path) -> str:
    """A path relative to the project root when possible (no local user names in committed reports)."""
    import os

    from app.core.config import PROJECT_ROOT

    try:
        return os.path.relpath(Path(path).resolve(), PROJECT_ROOT.resolve())
    except ValueError:      # another drive (Windows)
        return str(path)


def _snake(text: str) -> str:
    return "_".join(str(text).strip().lower().replace("-", " ").split())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _constraints(c: dict) -> dict:
    """Package query constraints -> the hard_constraints keys the chat evaluation checks."""
    out: dict[str, Any] = {}
    mapping = {"genres": "include_genres", "themes": "include_themes", "demographics": "include_demographics",
               "studios": "include_studios"}
    for k, v in (c or {}).items():
        if k in mapping:
            out[mapping[k]] = list(v) if isinstance(v, list) else [v]
        elif k == "type":
            out["types"] = [v] if isinstance(v, str) else list(v)
        elif k in ("min_score", "max_episodes", "min_episodes", "year_from", "year_to"):
            out[k] = v
        else:
            out.setdefault("unmapped", {})[k] = v
    return out


def _query_rows(path: Path, label_source: str) -> tuple[list[dict], Counter]:
    intents = {"scene": "scene_lookup", "semantic": "preference_recommendation",
               "constraint": "preference_recommendation"}
    rows, cats = [], Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        cats[q["category"]] += 1
        outcome = q.get("expected_outcome", "matches")
        rows.append({"query_id": f"team-{q['query_id']}", "query": q["query_en"], "category": q["category"],
                     "intent": intents.get(q["category"]),     # unanswerable queries have no single intent
                     "relevant_item_keys": [f"mal:{int(a)}" for a in q.get("expected_anime_ids") or []],
                     "expected_scene_ids": list(q.get("expected_scene_ids") or []),
                     "hard_constraints": _constraints(q.get("constraints") or {}),
                     "answerable": outcome == "matches", "expected_outcome": outcome,
                     "label_source": label_source, "note": q.get("evidence_note")})
    return rows, cats


def _scene_rows(path: Path, version: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        srcs = _json_field(s.get("sources")) or []
        first = srcs[0] if srcs else {}
        checked = f"checked {str(s.get('verified_at') or '')[:10]} by {s.get('verified_by') or 'unknown'} " \
                  f"({s.get('verification_method') or 'unknown method'})"
        ref = "; ".join(f"{x.get('title') or x.get('url')}" for x in srcs[:2] if isinstance(x, dict)) or "team package"
        title = (s.get("title_en") or "").strip()
        desc = (s.get("description_en") or "").strip()
        rows.append({"scene_id": s["scene_id"], "item_key": f"mal:{int(s['anime_id'])}",
                     "description": f"{title}. {desc}" if title and not desc.startswith(title) else (desc or title),
                     "episode": s.get("episode"), "timestamp_seconds": s.get("timestamp_seconds"),
                     "characters": list(s.get("characters") or []),
                     "source_ref": f"{ref} | {checked} | team package {version}",
                     "source_url": first.get("url") if isinstance(first, dict) else None,
                     "spoiler_level": "major",
                     "verification_status": "verified" if s.get("verification_status") == "verified" else "unverified"})
    return rows


def import_package(settings: Settings, package_dir: str | Path, force: bool = False, log=print) -> dict:
    t0 = time.time()
    pkg = Path(package_dir).expanduser()
    index_path = pkg / INDEX_FILE
    if not index_path.exists():
        raise PackageError(f"{index_path} not found (expected the team's data package root)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("catalog_status") != "final":
        raise PackageError(f"package catalog_status is {index.get('catalog_status')!r}, expected 'final'")
    version = str(index.get("catalog_version"))
    label_source = f"team_package_{version}"
    for rel in (CATALOG_FILE, SCENES_FILE, QUERIES_FILE):
        if not (pkg / rel).exists():
            raise PackageError(f"missing package file {rel}")

    enriched = settings._abs(settings.real.get("enriched_dir", "data/enriched"))
    evaluation = settings.data_root / "evaluation"
    rag_out = evaluation / "rag_queries.jsonl"
    if rag_out.exists() and not force:
        existing = {json.loads(line).get("label_source") for line in rag_out.read_text(encoding="utf-8").splitlines()
                    if line.strip()}
        if not existing or any(not str(s or "").startswith("team_package_") for s in existing):
            raise PackageError(f"{rag_out} exists and was not written by a package import; use --force to replace")

    tags, synopses, relations, characters = [], [], [], []
    counts: Counter = Counter()
    demo_reduced = []
    with (pkg / CATALOG_FILE).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            counts["records"] += 1
            if r.get("identity_status") != "confirmed":
                counts["skipped_identity_unresolved"] += 1
                continue
            mid = int(r["anime_id"])
            prov = _json_field(r.get("field_provenance")) or {}
            ref = f"team-package:{version}:anime_all#{mid}"
            themes = list(dict.fromkeys(t for t in (r.get("themes") or []) if t))
            src = "+".join(_sources(prov, "themes")) or "team_package"
            for t in themes:
                tags.append({"mal_id": mid, "namespace": "theme", "value": t, "source": src,
                             "source_ref": f"{ref}/themes"})
            demos = sorted(dict.fromkeys(d for d in (r.get("demographics") or []) if d))
            if demos:
                if len(demos) > 1:
                    demo_reduced.append({"mal_id": mid, "package": demos, "kept": demos[0]})
                tags.append({"mal_id": mid, "namespace": "demographic", "value": demos[0],
                             "source": "+".join(_sources(prov, "demographics")) or "team_package",
                             "source_ref": f"{ref}/demographics"})
            syn_src = _sources(prov, "synopsis_en")
            if r.get("synopsis_en") and syn_src:
                synopses.append({"mal_id": mid, "synopsis": r["synopsis_en"], "source": "+".join(syn_src),
                                 "source_ref": f"{ref}/synopsis_en"})
            rel_src = "+".join(_sources(prov, "relations")) or "team_package"
            for e in _json_field(r.get("relations")) or []:
                if isinstance(e, dict) and e.get("target_anime_id") and e.get("relation_type"):
                    relations.append({"from_mal_id": mid, "to_mal_id": int(e["target_anime_id"]),
                                      "relation_type": _snake(e["relation_type"]),
                                      "source_ref": f"{ref}/relations ({rel_src})"})
            chr_src = "+".join(_sources(prov, "characters")) or "team_package"
            for c in _json_field(r.get("characters")) or []:
                if isinstance(c, dict) and c.get("name") and c.get("source_character_id") is not None:
                    characters.append({"mal_id": mid, "name": c["name"], "role": c.get("role"),
                                       "character_source_id": f"{c.get('entity_namespace') or 'unknown'}:"
                                                              f"{c['source_character_id']}",
                                       "source_ref": f"{ref}/characters ({chr_src})"})
    relations = list({(x["from_mal_id"], x["to_mal_id"], x["relation_type"]): x for x in relations}.values())
    characters = list({(x["mal_id"], x["character_source_id"]): x for x in characters}.values())
    scenes = _scene_rows(pkg / SCENES_FILE, version)
    queries, cats = _query_rows(pkg / QUERIES_FILE, label_source)

    outputs = {"tags.jsonl": tags, "synopses.jsonl": synopses, "relations.jsonl": relations,
               "characters.jsonl": characters, "scenes.jsonl": scenes}
    for name, rows in outputs.items():
        _write_jsonl(enriched / name, rows)
    _write_jsonl(rag_out, queries)
    report = {
        "imported_at": utc_now_iso(), "package_dir": _display_path(pkg), "catalog_version": version,
        "package_index": {k: index.get(k) for k in ("schema_version", "catalog_status", "counts", "rag", "knowledge")},
        "files_read": {rel: sha256_file(pkg / rel) for rel in (INDEX_FILE, CATALOG_FILE, SCENES_FILE, QUERIES_FILE)},
        "files_not_read": NOT_READ,
        "written": {**{f"data/enriched/{k}": len(v) for k, v in outputs.items()},
                    "data/evaluation/rag_queries.jsonl": len(queries)},
        "catalog_records": dict(counts),
        "titles_with": {"themes": len({t["mal_id"] for t in tags if t["namespace"] == "theme"}),
                        "demographics": len({t["mal_id"] for t in tags if t["namespace"] == "demographic"}),
                        "characters": len({c["mal_id"] for c in characters}),
                        "relations": len({x["from_mal_id"] for x in relations}),
                        "synopses": len(synopses)},
        "query_categories": dict(cats),
        "policies": {
            "core_fields": "genres, studios, episodes, durations and scores are not imported (2023 snapshot stays "
                           "the single source; the frozen algorithm-2 model is bound to them)",
            "demographics": "one value per title: the alphabetically first (rule of the algorithm-1 projection "
                            "used to train the frozen algorithm-2 model)",
            "scenes": "verification status as delivered; checker and method recorded in source_ref; "
                      "spoiler_level defaults to major",
            "evaluation_queries": "label_source " + label_source + "; never indexed"},
        "demographics_reduced": demo_reduced,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(enriched / "team_package_import.json", report)
    log(f"[import-package] {version}: " + ", ".join(f"{k}={v}" for k, v in report["written"].items()))
    return report


def import_similarity_labels(settings: Settings, path: str | Path, log=print) -> dict:
    """Convert the algorithm-1 benchmark's MAL-recommendation ground truth into similarity_labels_mal.jsonl.

    Weak labels (MAL users' "if you liked A" recommendations), never human annotation. The benchmark's own
    pre-registered split is kept: historical -> train, validation -> val, test -> test.
    """
    src = Path(path)
    if not src.is_absolute():
        src = settings.data_root.parent / src
    if not src.exists():
        raise PackageError(f"{src} not found")
    split_map = {"historical": "train", "validation": "val", "test": "test"}
    rows, counts = [], Counter()
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        split = split_map.get(g.get("split"))
        if split is None or not g.get("relevant_ids"):
            counts["skipped"] += 1
            continue
        counts[split] += 1
        votes = list(g.get("votes") or [])
        rows.append({"query_item": f"mal:{int(g['target_id'])}", "split": split,
                     "positives": [{"item_key": f"mal:{int(a)}", "grade": 1,
                                    **({"votes": votes[k]} if k < len(votes) else {})}
                                   for k, a in enumerate(g["relevant_ids"])],
                     "label_source": "mal_user_recommendations_weak", "source_url": g.get("source_url"),
                     "note": f"algorithm1_benchmark ground truth ({g.get('sampling_group')})"})
    out = settings.data_root / "evaluation" / "similarity_labels_mal.jsonl"
    _write_jsonl(out, rows)
    report = {"from": str(path), "sha256": sha256_file(src), "rows": len(rows), "splits": dict(counts),
              "label_source": "mal_user_recommendations_weak", "written": str(out.relative_to(settings.data_root.parent))}
    log(f"[import-similarity-labels] {report['rows']} queries {dict(counts)} -> {report['written']}")
    return report
