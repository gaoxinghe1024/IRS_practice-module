"""Contract validation for optional supplementary and evaluation files (see docs/data-contracts.md)."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.core.ids import is_valid_item_key

CONTRACTS: dict[str, dict] = {
    "enriched/relations.jsonl": {"required": ["from_mal_id", "relation_type", "source_ref"],
                                 "one_of": [["to_mal_id", "to_external_id"]]},
    "enriched/characters.jsonl": {"required": ["mal_id", "character_source_id", "name", "source_ref"]},
    "enriched/reviews.jsonl": {"required": ["mal_id", "review_id", "text", "source_ref"]},
    "enriched/tags.jsonl": {"required": ["mal_id", "namespace", "value", "source_ref"],
                            "enums": {"namespace": ["genre", "theme", "demographic", "studio"]}},
    "enriched/synopses.jsonl": {"required": ["mal_id", "synopsis", "source_ref"]},
    "enriched/scenes.jsonl": {"required": ["scene_id", "item_key", "description", "source_ref", "verification_status"],
                              "item_key_fields": ["item_key"],
                              "enums": {"verification_status": ["verified", "unverified"],
                                        "spoiler_level": ["none", "minor", "major"]}},
    "enriched/scene_images/manifest.jsonl": {"required": ["image_id", "item_key", "path", "sha256", "source_ref"],
                                             "item_key_fields": ["item_key"]},
    "evaluation/similarity_labels.jsonl": {"required": ["query_item", "positives", "label_source"],
                                           "item_key_fields": ["query_item"]},
    "evaluation/similarity_labels_mal.jsonl": {"required": ["query_item", "positives", "label_source"],
                                               "item_key_fields": ["query_item"],
                                               "enums": {"split": ["train", "val", "test"]}},
    "evaluation/collection_events.jsonl": {"required": ["collection_id", "offline_user_id", "items"]},
    # intent may be null for a query with no single expected intent (e.g. a request the data cannot answer)
    "evaluation/rag_queries.jsonl": {"required": ["query_id", "query", "intent", "relevant_item_keys", "answerable"],
                                     "nullable": ["intent"],
                                     "enums": {"expected_outcome": ["matches", "no_match", "insufficient_information"]}},
    "evaluation/scene_queries.jsonl": {"required": ["query_id", "query", "target_item_key"]},
    "evaluation/image_queries.jsonl": {"required": ["query_id", "path", "sha256", "in_gallery"]},
    "evaluation/system_tasks.jsonl": {"required": ["task_id", "surface", "steps", "success"]},
}


def validate_file(path: Path, contract: dict) -> dict:
    errors, n = [], 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            n += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
                continue
            nullable = contract.get("nullable", [])
            missing = [f for f in contract.get("required", [])
                       if f not in row or (row[f] in (None, "") and f not in nullable)]
            if missing:
                errors.append(f"line {line_no}: missing {missing}")
            for group in contract.get("one_of", []):
                if not any(row.get(f) not in (None, "") for f in group):
                    errors.append(f"line {line_no}: needs one of {group}")
            for f in contract.get("item_key_fields", []):
                if row.get(f) and not is_valid_item_key(str(row[f])):
                    errors.append(f"line {line_no}: {f}={row[f]!r} is not an item key (mal:<id> / synthetic:<seed>:<n>)")
            for f, allowed in contract.get("enums", {}).items():
                if f in row and row[f] not in allowed:
                    errors.append(f"line {line_no}: {f} must be one of {allowed}")
            if len(errors) > 50:
                errors.append("... (truncated)")
                break
    return {"rows": n, "ok": not errors, "errors": errors[:51]}


def validate_all(settings: Settings) -> dict:
    root = settings.data_root
    out = {}
    for rel_path, contract in CONTRACTS.items():
        p = root / rel_path
        out[rel_path] = validate_file(p, contract) if p.exists() else {"present": False}
    return out
