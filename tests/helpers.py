"""Small in-memory catalogue builders for unit tests (fictional items)."""

from __future__ import annotations

import pandas as pd

from app.recommendation.features import build_snapshot


def make_snapshot(rows: list[dict], version: str = "test-v1"):
    base = {"mal_id": None, "title_english": None, "title_other": None, "synopsis": None, "score": 7.0,
            "type": "TV", "episodes": 12, "duration_minutes": 24.0, "total_minutes": 288.0, "status": "Finished Airing",
            "members": 100, "aired_from": "2010", "in_catalog": True, "is_recommendable": True,
            "is_planner_eligible": True, "is_explicit": False}
    cat, tags = [], []
    for n, r in enumerate(rows):
        key = r.get("item_key", f"synthetic:1:{n + 1:03d}")
        rec = {**base, **{k: v for k, v in r.items() if k not in ("genre", "theme", "demographic", "studio")}}
        rec["item_key"] = key
        rec.setdefault("title", f"Item {n + 1}")
        if rec["episodes"] is not None and rec["duration_minutes"] is not None:
            rec["total_minutes"] = rec["episodes"] * rec["duration_minutes"]
        else:
            rec["total_minutes"] = None
        cat.append(rec)
        for ns in ("genre", "theme", "demographic", "studio"):
            for v in r.get(ns, []):
                tags.append({"item_key": key, "namespace": ns, "value": v, "source": "test"})
    return build_snapshot(pd.DataFrame(cat), pd.DataFrame(tags), version, "synthetic")
