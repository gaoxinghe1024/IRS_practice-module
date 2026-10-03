# -*- coding: utf-8 -*-
"""test_replay.py — analyze.py replay output vs the packaged tables.

Fast mode (200 bootstrap resamples): deterministic columns must match exactly
(floats compared with exact equality after rounding to 6 decimals); bootstrap CI
columns are only structural here. The full 10,000-resample run was executed once
and its per-file diff recorded in the workspace audit record."""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
PY = sys.executable


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="replay_test_")
        r = subprocess.run(
            [PY, str(PKG / "harness" / "analyze.py"), "--fast",
             "--out-dir", cls.tmp],
            capture_output=True, text=True, cwd=str(PKG), timeout=280)
        if r.returncode != 0:
            raise AssertionError(f"analyze.py failed:\n{r.stdout}\n{r.stderr}")
        cls.summary = {f'{r["model_label"]}|{r["group"]}': r
                       for r in read_csv(Path(cls.tmp) / "replay_summary.csv")}
        cls.latency = {r["model_label"]: r
                       for r in read_csv(Path(cls.tmp) / "replay_latency.csv")}
        cls.tokens = {r["model_label"]: r
                      for r in read_csv(Path(cls.tmp) / "replay_token_usage.csv")}

    def fnum(self, x):
        if x in ("", None):
            return None
        if x in ("True", "true"):
            return 1.0
        if x in ("False", "false"):
            return 0.0
        return float(x)

    def close(self, a, b, tol=1e-9):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(a - b) <= tol

    def test_summary_matches_scored_csv(self):
        scored = read_csv(PKG / "results" / "scored.csv")
        by: dict[tuple, list[dict]] = {}
        for r in scored:
            by.setdefault((r["model_label"], r["group"]), []).append(r)
        for key, rows in by.items():
            rep = self.summary[f"{key[0]}|{key[1]}"]
            self.assertEqual(int(rep["n"]), len(rows), key)
            for metric in ("judge_avg", "episode_ok", "all_satisfy", "fact_ok",
                           "rule_frank", "in_kb_reply_ok", "top1"):
                vals = [self.fnum(r.get(metric)) for r in rows]
                vals = [v for v in vals if v is not None]
                m = sum(vals) / len(vals) if vals else None
                self.assertTrue(self.close(self.fnum(rep.get(metric)), m),
                                f"{key}.{metric}")

    def test_latency_matches_published_one_decimal(self):
        # published values (report, one decimal, linear interpolation) vs replay
        published = {"Deepseek-V4.1-Flash": 8.1, "GLM-5.3": 15.6, "Hy3": 19.1,
                     "Kimi-K3": 19.7, "GLM-5.3-Flash": 20.5, "MiniMax-M3": 32.4}
        for ml, want in published.items():
            got = round(self.fnum(self.latency[ml]["p95_linear_s"]), 1)
            self.assertEqual(got, want, ml)

    def test_over20s_counts(self):
        counts = {"Deepseek-V4.1-Flash": 0, "GLM-5.3": 2, "GLM-5.3-Flash": 6,
                  "Hy3": 4, "Kimi-K3": 3, "MiniMax-M3": 12}
        for ml, want in counts.items():
            self.assertEqual(int(self.latency[ml]["over_20s"]), want, ml)

    def test_token_levels_match_report(self):
        # R-group output tokens per call (absolute levels, mean)
        published = {"Kimi-K3": 277, "GLM-5.3": 305, "GLM-5.3-Flash": 313,
                     "MiniMax-M3": 528, "Deepseek-V4.1-Flash": 628, "Hy3": 790}
        for ml, want in published.items():
            got = round(self.fnum(self.tokens[ml]["R_output_tokens"]))
            self.assertEqual(got, want, ml)

    def test_rag_lift_deterministic_means(self):
        lift = read_csv(Path(self.tmp) / "replay_rag_lift.csv")
        by = {r["model_label"]: r for r in lift}
        # judge lift means published in the report: +0.41..+0.67 overall range
        vals = [self.fnum(r["judge_avg"]) for r in lift]
        self.assertTrue(all(v is not None for v in vals))
        self.assertLessEqual(min(vals), 0.45)
        self.assertGreaterEqual(max(vals), 0.60)
        # fact lift must be negative for every model
        for ml, r in by.items():
            fv = self.fnum(r["fact_ok"])
            self.assertIsNotNone(fv)
            self.assertLess(fv, -0.4, ml)


if __name__ == "__main__":
    unittest.main()
