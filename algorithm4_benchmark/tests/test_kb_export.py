# -*- coding: utf-8 -*-
"""test_kb_export.py — data/kb/anime_kb_export.json must be field-for-field
identical to data/kb/anime.db (tables: anime 24 rows, scenes 12 rows).
The DB is the runtime source of truth; the JSON is the readable export."""
from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DB = PKG / "data" / "kb" / "anime.db"
JS = PKG / "data" / "kb" / "anime_kb_export.json"


class TestKbExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        con = sqlite3.connect(str(DB))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cls.db_anime = [dict(r) for r in cur.execute(
            "SELECT * FROM anime ORDER BY id")]
        cls.db_scenes = [dict(r) for r in cur.execute(
            "SELECT * FROM scenes ORDER BY id")]
        con.close()
        cls.export = json.loads(JS.read_text(encoding="utf-8"))

    def test_shape(self):
        self.assertEqual(len(self.db_anime), 24)
        self.assertEqual(len(self.db_scenes), 12)
        self.assertEqual(set(self.export["tables"].keys()), {"anime", "scenes"})
        self.assertIn("source", self.export)

    def test_anime_field_for_field(self):
        self.assertEqual(self.export["tables"]["anime"], self.db_anime,
                         "anime table differs from anime.db")

    def test_scenes_field_for_field(self):
        self.assertEqual(self.export["tables"]["scenes"], self.db_scenes,
                         "scenes table differs from anime.db")

    def test_database_holds_only_the_two_kb_tables(self):
        con = sqlite3.connect(str(DB))
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        con.close()
        self.assertEqual(names, {"anime", "scenes"})

    def test_no_user_tables_leaked(self):
        blob = JS.read_text(encoding="utf-8")
        self.assertNotIn("password_hash", blob)
        self.assertNotIn("sessions", blob)


if __name__ == "__main__":
    unittest.main()
