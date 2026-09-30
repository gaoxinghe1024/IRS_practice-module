"""Legacy account migration: validation, reporting, dry-run safety and title-id verification."""

from __future__ import annotations

import hashlib
import sqlite3

from sqlalchemy import func, select

from app.db.models import Collection, Rating, User
from app.db.session import session_scope
from evaluation.sandbox import sandbox_settings
from pipelines.legacy_migration import TitleResolver, legacy_mal_table, migrate


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class _FakeDB:
    """Answers TitleResolver's two queries (catalogue rows, then aliases) in order."""

    def __init__(self, anime, aliases):
        self.answers = [anime, aliases]

    def execute(self, _stmt):
        return _Rows(self.answers.pop(0))


def test_resolver_verifies_legacy_ids_against_catalogue_titles():
    anime = [("mal:35860", "Karakai Jouzu no Takagi-san", "Teasing Master Takagi-san"),
             ("mal:37141", "Hataraku Saibou", "Cells at Work!"),
             ("mal:39586", "Hataraku Saibou!!", "Cells at Work!!"),
             ("mal:1535", "Death Note", "Death Note"),
             ("mal:1", "Twin", None), ("mal:2", "Twin", None)]
    table = {"Hataraku Saibou": 35860, "Death Note": 1535, "Twin": 3}
    r = TitleResolver(_FakeDB(anime, [("mal:1535", "デスノート")]), table)
    assert r.resolve("Death Note", None) == ("mal:1535", None)
    assert r.resolve("Hataraku Saibou", None) == ("mal:37141", None)           # wrong legacy id corrected ...
    assert r.corrections and r.corrections[0]["legacy_mal_id"] == 35860       # ... and reported
    key, why = r.resolve("Twin", None)                                         # ambiguous exact title: no guess
    assert key is None and "not in the target catalogue" in why
    key, why = r.resolve("L: Change the World", None)
    assert key is None and "no MAL id" in why


def test_shipped_seed_table_is_the_legacy_table():
    table = legacy_mal_table()
    assert len(table) == 23 and table["Death Note"] == 1535 and "L: Change the World" not in table


def _legacy_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, gender TEXT DEFAULT 'Unknown',
                            mean_score REAL DEFAULT 0, password_hash TEXT DEFAULT '', created_at TEXT DEFAULT '');
        CREATE TABLE anime (id INTEGER PRIMARY KEY, title TEXT, mal_id INTEGER DEFAULT 0);
        CREATE TABLE ratings (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, anime_id INTEGER, rating REAL,
                              created_at TEXT DEFAULT '', UNIQUE(user_id, anime_id));
        CREATE TABLE favorites (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, anime_id INTEGER,
                                folder_id INTEGER DEFAULT 0, UNIQUE(user_id, anime_id));
        CREATE TABLE fav_folders (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, created_at TEXT);
    """)
    good_hash = "0123456789abcdef$" + hashlib.pbkdf2_hmac("sha256", b"legacy pw", b"0123456789abcdef", 60000).hex()
    con.executemany("INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                    [(1, "demo", ""), (2, "legacy_ok", good_hash), (3, "bad hash", "plain"),
                     (4, "legacy_badfmt", "not-a-pbkdf2-hash")])
    con.executemany("INSERT INTO anime (id, title) VALUES (?, ?)", [(1, "Death Note"), (2, "L: Change the World")])
    con.executemany("INSERT INTO ratings (id, user_id, anime_id, rating, created_at) VALUES (?, ?, ?, ?, ?)",
                    [(1, 2, 1, 8.5, "2026-08-01 19:00:00"),       # seed back-fill pattern for id 1
                     (2, 2, 2, 0.2, "2026-09-18 10:00:00")])      # invalid value
    con.execute("INSERT INTO fav_folders (id, user_id, name) VALUES (1, 2, 'Rewatch')")
    con.executemany("INSERT INTO favorites (user_id, anime_id, folder_id) VALUES (?, ?, ?)", [(2, 1, 1), (2, 2, 0)])
    con.commit()
    con.close()


def test_dry_run_reports_everything_and_changes_nothing(synthetic_env, tmp_path):
    legacy = tmp_path / "anime.db"
    _legacy_db(legacy)
    with sandbox_settings(synthetic_env) as sb:
        with session_scope(sb.db_url) as db:
            before = db.scalar(select(func.count(User.id)))
        rep = migrate(sb, legacy, apply=False, log=lambda *_: None)
        with session_scope(sb.db_url) as db:
            assert db.scalar(select(func.count(User.id))) == before              # dry run wrote nothing
        reasons = {u["legacy_user"]: u["reason"] for u in rep["skipped_users"]}
        assert all("ratings_not_migrated" in u and "favorites_not_migrated" in u for u in rep["skipped_users"])
        assert reasons == {"demo": "no password (seed/demo user)", "bad hash": "username violates the new rules",
                           "legacy_badfmt": "unrecognised password hash format"}
        assert [u["legacy_user"] for u in rep["users"]] == ["legacy_ok"]
        assert rep["invalid_ratings"][0]["value"] == 0.2
        # synthetic catalogues have no MAL ids: the rating and both favourites are listed, not dropped silently
        kinds = sorted(u["kind"] for u in rep["unmapped"])
        assert kinds == ["favorite", "favorite", "rating"]
        assert rep["legacy_counts"] == {"users": 4, "anime": 2, "ratings": 2, "favorites": 2, "fav_folders": 1}
        assert rep["orphan_rows"] == {"ratings": 0, "favorites": 0}

        applied = migrate(sb, legacy, apply=True, log=lambda *_: None)
        assert applied["backup"] and applied["users"][0]["status"] == "migrated"
        with session_scope(sb.db_url) as db:
            u = db.scalar(select(User).where(User.normalized_username == "legacy_ok"))
            assert u is not None and u.origin == "legacy_migration"
            assert db.scalar(select(func.count(Rating.id)).where(Rating.user_id == u.id)) == 0
            assert db.scalar(select(func.count(Collection.id)).where(Collection.user_id == u.id)) == 0
        again = migrate(sb, legacy, apply=False, log=lambda *_: None)                # idempotent: no duplicate
        assert again["users"][0]["status"] == "username exists in target - skipped"
