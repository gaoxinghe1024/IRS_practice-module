"""Migrate user data from the legacy template database (backend/anime.db).

Dry-run by default: validates every row, writes a report and changes nothing (the target database
must already exist, i.e. a dataset has been activated). With --apply it first copies the target
database to a timestamped backup. The legacy database is always opened read-only.

Mapping rules
  * users: only accounts with a password hash (demo/visitor seed users have none); the legacy
    PBKDF2 hash (``<16 hex salt>$<64 hex digest>``) is kept in a verifiable form and upgraded to
    argon2id at first login. Other hash formats and invalid usernames are reported and skipped.
    Legacy gender / mean-score columns are not imported.
  * titles: the legacy catalogue is a hand-written 24-title seed whose own mal_id column is 0; titles
    are mapped through pipelines/legacy_seed_mal_ids.json (copied from the legacy template). Every id is
    checked against the title of that MAL id in the target catalogue; on a mismatch the title is
    resolved only by a unique exact title match, and the correction is reported. Titles only resolve
    in REAL mode (synthetic items have no MAL ids). Unmapped rows are reported, never dropped silently.
  * ratings: legacy floats (e.g. 8.5) are rounded half-up to integers 1-10; every conversion is
    reported; values outside 0.5-10 are reported as invalid and not migrated.
  * favourites / folders -> collections ("Imported favourites" for the default folder).
  * provenance flag: ratings whose timestamps follow the legacy seed back-fill pattern
    ('2026-08-01 12:00 + id*7 hours') were probably inherited from orphaned seed rows, not
    entered by the user; they are listed and migrated only with --include-seed-pattern.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings
from app.core.security import LEGACY_PREFIX
from app.core.text import nfkc, norm_title
from app.db.models import Anime, AnimeAlias, Collection, CollectionItem, Rating, User
from app.db.session import run_migrations, session_scope
from app.services.auth import USERNAME_RE, normalize_username
from pipelines.common import write_json

SEED_TABLE = Path(__file__).with_name("legacy_seed_mal_ids.json")
LEGACY_HASH_RE = re.compile(r"^[0-9a-f]{16}\$[0-9a-f]{64}$")


def legacy_mal_table(path: Path = SEED_TABLE) -> dict[str, int]:
    return {str(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8"))["seed_mal_ids"].items()}


def _seed_pattern(rating_id: int, created_at: str | None) -> bool:
    if not created_at:
        return False
    try:
        ts = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return ts == datetime(2026, 8, 1, 12, 0) + timedelta(hours=7 * rating_id)


def _fold(title: str) -> str:
    return " ".join(nfkc(title).casefold().split())


class TitleResolver:
    """Legacy title -> target item_key, verified against the target catalogue."""

    def __init__(self, db, table: dict[str, int]):
        self.table = table
        self.names: dict[str, set[str]] = {}          # item_key -> normalised names (title, English, aliases)
        self.by_exact: dict[str, set[str]] = {}       # case-folded title (punctuation kept) -> item_keys
        self.by_name: dict[str, set[str]] = {}        # normalised title (punctuation removed) -> item_keys
        for key, title, eng in db.execute(select(Anime.item_key, Anime.title, Anime.title_english)
                                          .where(Anime.in_catalog.is_(True))):
            for name in (title, eng):
                if name:
                    self.names.setdefault(key, set()).add(norm_title(name))
            self.by_exact.setdefault(_fold(title), set()).add(key)
            self.by_name.setdefault(norm_title(title), set()).add(key)
        for key, alias in db.execute(select(AnimeAlias.item_key, AnimeAlias.alias)):
            if key in self.names:
                self.names[key].add(norm_title(alias))
        self.corrections: list[dict] = []
        self._cache: dict[str, tuple[str | None, str | None]] = {}

    def resolve(self, title: str | None, legacy_mal_id: int | None) -> tuple[str | None, str | None]:
        """(item_key or None, reason when None)."""
        if not title:
            return None, "legacy row without a title"
        if title in self._cache:
            return self._cache[title]
        mid = legacy_mal_id or self.table.get(title)
        want = norm_title(title)
        out: tuple[str | None, str | None]
        if mid and want in self.names.get(f"mal:{mid}", set()):
            out = (f"mal:{mid}", None)
        else:
            exact = sorted(self.by_exact.get(_fold(title), set()))
            if len(exact) != 1:                        # "Hataraku Saibou" must not collide with "Hataraku Saibou!!"
                exact = sorted(self.by_name.get(want, set()))
            if len(exact) == 1:
                out = (exact[0], None)
                if mid:
                    self.corrections.append({"legacy_title": title, "legacy_mal_id": mid, "resolved_key": exact[0],
                                             "evidence": "unique exact title match in the target catalogue; the "
                                                         "legacy id names a different title"})
            elif not mid:
                out = (None, "no MAL id for this title (not an anime in MAL or not in the legacy table)")
            elif f"mal:{mid}" not in self.names:
                out = (None, f"mal:{mid} is not in the target catalogue")
            else:
                out = (None, f"legacy id mal:{mid} names a different title and no unique exact title match exists")
        self._cache[title] = out
        return out


def migrate(settings: Settings, legacy_db: Path, apply: bool = False, include_seed_pattern: bool = False,
            log=print) -> dict:
    legacy_db = Path(legacy_db)
    if not legacy_db.exists():
        raise FileNotFoundError(f"legacy database not found: {legacy_db}")
    if not settings.db_path.exists():
        raise FileNotFoundError(f"target database not found: {settings.db_path} (activate a dataset first)")
    src = sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    report = {"mode": settings.mode, "apply": apply, "legacy_db": str(legacy_db), "include_seed_pattern":
              include_seed_pattern, "users": [], "skipped_users": [], "unmapped": [], "title_corrections": [],
              "rating_conversions": [], "invalid_ratings": [], "seed_pattern_ratings": [], "backup": None,
              "legacy_counts": {t: src.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                                for t in ("users", "anime", "ratings", "favorites", "fav_folders") if t in tables}}
    # rows whose user no longer exists (the legacy seed generated rating rows for users it never created)
    report["orphan_rows"] = {t: src.execute(f'SELECT COUNT(*) FROM "{t}" WHERE user_id NOT IN '
                                            '(SELECT id FROM users)').fetchone()[0]
                             for t in ("ratings", "favorites") if t in tables}
    anime = {r["id"]: dict(r) for r in src.execute("SELECT id, title, mal_id FROM anime")}
    if apply:
        backup = settings.db_path.with_suffix(f".backup-{time.strftime('%Y%m%d-%H%M%S')}.sqlite3")
        shutil.copyfile(settings.db_path, backup)
        report["backup"] = str(backup)
        run_migrations(settings.db_url)
    with session_scope(settings.db_url) as db:
        resolver = TitleResolver(db, legacy_mal_table())

        def key_for(aid: int) -> tuple[str | None, str | None, str | None]:
            a = anime.get(aid) or {}
            key, why = resolver.resolve(a.get("title"), a.get("mal_id") or None)
            return key, why, a.get("title")

        def skip(u, reason: str) -> None:
            # a skipped account's rows are counted, so nothing disappears from the report
            n_r = src.execute("SELECT COUNT(*) FROM ratings WHERE user_id=?", (u["id"],)).fetchone()[0]
            n_f = src.execute("SELECT COUNT(*) FROM favorites WHERE user_id=?", (u["id"],)).fetchone()[0] \
                if "favorites" in tables else 0
            report["skipped_users"].append({"legacy_user": u["username"] or "", "reason": reason,
                                            "ratings_not_migrated": n_r, "favorites_not_migrated": n_f})

        for u in src.execute("SELECT id, username, password_hash, created_at FROM users").fetchall():
            name = u["username"] or ""
            if not u["password_hash"]:
                skip(u, "no password (seed/demo user)")
                continue
            if not USERNAME_RE.match(name):
                skip(u, "username violates the new rules")
                continue
            if not LEGACY_HASH_RE.match(u["password_hash"]):
                skip(u, "unrecognised password hash format")
                continue
            norm = normalize_username(name)
            entry = {"legacy_user": name, "ratings": 0, "favorites": 0, "collections": 0, "unmapped": 0,
                     "seed_pattern_skipped": 0, "invalid": 0}
            if db.scalar(select(User.id).where(User.normalized_username == norm)):
                entry["status"] = "username exists in target - skipped"
                report["users"].append(entry)
                continue
            user = None
            if apply:
                user = User(normalized_username=norm, display_name=name,
                            password_hash=LEGACY_PREFIX + u["password_hash"], origin="legacy_migration",
                            onboarding_state="done")
                db.add(user)
                db.flush()
            for r in src.execute("SELECT id, anime_id, rating, created_at FROM ratings WHERE user_id=?", (u["id"],)):
                key, why, title = key_for(r["anime_id"])
                try:
                    raw = float(r["rating"])
                except (TypeError, ValueError):
                    raw = float("nan")
                if not math.isfinite(raw) or not 0.5 <= raw <= 10:
                    entry["invalid"] += 1
                    report["invalid_ratings"].append({"legacy_user": name, "legacy_title": title, "value": r["rating"]})
                    continue
                if key is None:
                    entry["unmapped"] += 1
                    report["unmapped"].append({"legacy_user": name, "kind": "rating", "legacy_title": title,
                                               "reason": why})
                    continue
                if _seed_pattern(r["id"], r["created_at"]):
                    report["seed_pattern_ratings"].append({"legacy_user": name, "legacy_title": title,
                                                           "created_at": r["created_at"],
                                                           "migrated": include_seed_pattern})
                    if not include_seed_pattern:
                        entry["seed_pattern_skipped"] += 1
                        continue
                val = int(min(10, max(1, math.floor(raw + 0.5))))
                if val != raw:
                    report["rating_conversions"].append({"legacy_user": name, "from": r["rating"], "to": val,
                                                         "item_key": key})
                if apply:
                    db.add(Rating(user_id=user.id, item_key=key, rating=val))
                entry["ratings"] += 1
            folders = {r["id"]: r["name"] for r in src.execute("SELECT id, name FROM fav_folders WHERE user_id=?",
                                                              (u["id"],))} if "fav_folders" in tables else {}
            colls: dict[int, Collection] = {}
            placed: set[tuple[int, str]] = set()
            favs = src.execute("SELECT anime_id, folder_id FROM favorites WHERE user_id=?", (u["id"],)) \
                if "favorites" in tables else []
            for f in favs:
                key, why, title = key_for(f["anime_id"])
                fid = f["folder_id"] or 0
                if fid and fid not in folders:
                    fid = 0                                   # dangling folder id -> default collection
                if key is None:
                    entry["unmapped"] += 1
                    report["unmapped"].append({"legacy_user": name, "kind": "favorite", "legacy_title": title,
                                               "folder": folders.get(fid, "Imported favourites"), "reason": why})
                    continue
                if (fid, key) in placed:
                    continue
                placed.add((fid, key))
                if fid not in colls:
                    entry["collections"] += 1
                    colls[fid] = Collection(user_id=user.id, name=folders.get(fid, "Imported favourites")[:60]) \
                        if apply else None
                    if apply:
                        db.add(colls[fid])
                        db.flush()
                if apply:
                    db.add(CollectionItem(collection_id=colls[fid].id, item_key=key))
                entry["favorites"] += 1
            entry["status"] = "migrated" if apply else "would migrate"
            report["users"].append(entry)
        report["title_corrections"] = resolver.corrections
        if not apply:
            db.rollback()
    out = settings.var_dir / f"legacy_migration_{'apply' if apply else 'dryrun'}.json"
    write_json(out, report)
    log(f"[migrate-legacy] {'APPLIED' if apply else 'dry run'}: users={len(report['users'])} "
        f"skipped_users={len(report['skipped_users'])} unmapped={len(report['unmapped'])} "
        f"corrections={len(report['title_corrections'])} seed-pattern ratings={len(report['seed_pattern_ratings'])}"
        f" -> {out}")
    return report
