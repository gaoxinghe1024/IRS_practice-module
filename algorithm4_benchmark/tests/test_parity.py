# -*- coding: utf-8 -*-
"""test_parity.py — snapshot parity: rebuild retrieval / prompts / T answers and
compare byte-for-byte with the saved artifacts. All three must be 80/80."""
from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "algo"))

import prompts  # noqa: E402

RESULTS = PKG / "results"


def load_jsonl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testset = load_jsonl(PKG / "data" / "testset_v1.jsonl")
        cls.retrieval_saved = {x["id"]: x["retrieval"]
                               for x in load_jsonl(RESULTS / "retrieval.jsonl")}
        raw = load_jsonl(RESULTS / "raw.jsonl.gz")
        cls.t_saved = {r["qid"]: r for r in raw if r.get("group") == "T"}

    def test_01_retrieval_80_of_80(self):
        mism = []
        for q in self.testset:
            rebuilt = prompts.retrieve(q["query"])
            if json.dumps(rebuilt, ensure_ascii=False, sort_keys=True) != \
                    json.dumps(self.retrieval_saved[q["id"]], ensure_ascii=False,
                               sort_keys=True):
                mism.append(q["id"])
        self.assertEqual(mism, [], f"retrieval mismatches: {mism}")
        self.assertEqual(len(self.retrieval_saved), 80)

    def test_02_r_prompts_80_of_80(self):
        mism = []
        for q in self.testset:
            saved = (RESULTS / "extra" / "parity_prompts" / "R" / f"{q['id']}.txt") \
                .read_text(encoding="utf-8")
            rebuilt = prompts.build_r_prompt(self.retrieval_saved[q["id"]], q["query"])
            if rebuilt != saved:
                mism.append(q["id"])
        self.assertEqual(mism, [], f"R prompt mismatches: {mism}")

    def test_03_n_prompts_80_of_80(self):
        mism = []
        for q in self.testset:
            saved = (RESULTS / "extra" / "parity_prompts" / "N" / f"{q['id']}.txt") \
                .read_text(encoding="utf-8")
            rebuilt = prompts.build_n_prompt(q["query"])
            if rebuilt != saved:
                mism.append(q["id"])
        self.assertEqual(mism, [], f"N prompt mismatches: {mism}")

    def test_04_t_answers_80_of_80(self):
        mism = []
        for q in self.testset:
            rebuilt = prompts.t_answer(q["query"])["reply"]
            if rebuilt != (self.t_saved[q["id"]].get("reply") or ""):
                mism.append(q["id"])
        self.assertEqual(mism, [], f"T answer mismatches: {mism}")
        self.assertEqual(len(self.t_saved), 80)

    def test_05_t_answers_file_matches_raw_t_group(self):
        # results/extra/t_answers.jsonl.gz holds the same 80 template answers
        rows = load_jsonl(RESULTS / "extra" / "t_answers.jsonl.gz")
        self.assertEqual(len(rows), 80)
        self.assertEqual({r["id"] for r in rows}, set(self.t_saved))
        for r in rows:
            saved = self.t_saved[r["id"]]
            self.assertEqual(r["answer"], saved.get("reply") or "", r["id"])
            self.assertEqual(r["scene_focus_title"] or None,
                             saved.get("scene_focus_title") or None, r["id"])


if __name__ == "__main__":
    unittest.main()
