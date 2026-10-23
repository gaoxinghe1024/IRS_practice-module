"""Explicit demo accounts for SYNTHETIC mode only (`anilogic seed-demo`).

Each persona rates titles that match its taste (plus a few low ratings), saves
a collection and some progress, and deliberately leaves several same-taste
titles unrated so the recommenders have something to find. Passwords are
documented in docs/user-guide.md and only exist in the synthetic database.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select

from app.core.config import Settings
from app.core.text import norm
from app.db.models import User
from app.db.session import run_migrations, session_scope
from app.recommendation.features import build_snapshot
from app.services import auth, user_data

DEMO_PASSWORD = "anilogic-demo"
PERSONAS = {
    "demo_mystery": {"tags": [("genre", "Mystery"), ("genre", "Suspense"), ("theme", "Psychological"),
                              ("theme", "Detective")], "collection": "Mind games"},
    "demo_action": {"tags": [("genre", "Action"), ("genre", "Adventure"), ("theme", "Martial Arts"),
                             ("theme", "Super Power")], "collection": "Weekend battles"},
    "demo_healing": {"tags": [("genre", "Slice of Life"), ("theme", "Iyashikei"), ("theme", "Music"),
                              ("genre", "Romance")], "collection": "Comfort shows"},
    "demo_scifi": {"tags": [("genre", "Sci-Fi"), ("theme", "Mecha"), ("theme", "Space"), ("theme", "Time Travel")],
                   "collection": "Space and machines"},
}


def seed_demo(settings: Settings, log=print) -> dict:
    if settings.mode != "synthetic":
        raise RuntimeError("Demo accounts are only created in synthetic mode.")
    import json

    import pandas as pd

    from pipelines.common import read_json

    active = read_json(settings.artifacts_dir / "active.json")
    art = settings._abs(active["artifact_dir"])
    snap = build_snapshot(pd.read_parquet(art / "catalog" / "catalog.parquet"),
                          pd.read_parquet(art / "catalog" / "item_tags.parquet"), active["dataset_version"], "synthetic")
    run_migrations(settings.db_url)
    rng = np.random.default_rng(7)
    created = {}
    with session_scope(settings.db_url) as db:
        for name, spec in PERSONAS.items():
            if db.scalar(select(User.id).where(User.normalized_username == name)):
                created[name] = "exists"
                continue
            issued = auth.register(db, name, DEMO_PASSWORD, 1, origin="synthetic_demo_seed")
            user = issued.user
            user.onboarding_state = "done"
            match = np.zeros(snap.n)
            for ns, v in spec["tags"]:
                tm = snap.tags.get(ns)
                if tm and norm(v) in tm.col:
                    match += np.asarray(tm.X[:, tm.col[norm(v)]].toarray()).ravel()
            pool = np.flatnonzero(snap.recommendable & (match > 0))
            order = pool[np.argsort(-(match[pool] + rng.random(len(pool)) * 0.5))]
            liked, held_out = order[:10], order[10:14]
            for i in liked:
                user_data.put_rating(db, user, str(snap.keys[i]), int(rng.integers(8, 11)))
            others = np.flatnonzero(snap.recommendable & (match == 0))
            for i in rng.choice(others, size=min(2, len(others)), replace=False):
                user_data.put_rating(db, user, str(snap.keys[i]), int(rng.integers(3, 6)))
            coll = user_data.create_collection(db, user, spec["collection"], "Demo collection (synthetic)")
            for i in liked[:5]:
                user_data.add_collection_item(db, user, coll.id, str(snap.keys[i]))
            user_data.put_progress(db, user, str(snap.keys[liked[0]]), "completed", None)
            created[name] = {"ratings": len(liked) + 2, "held_out_same_taste": [str(snap.keys[i]) for i in held_out]}
        if not db.scalar(select(User.id).where(User.normalized_username == "demo_admin")):
            admin = auth.register(db, "demo_admin", DEMO_PASSWORD, 1, origin="synthetic_demo_seed").user
            admin.is_admin = True
            admin.onboarding_state = "done"
            created["demo_admin"] = "admin"
    log(f"[seed-demo] {json.dumps(created)[:400]}")
    return created
