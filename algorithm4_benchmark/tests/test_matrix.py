# -*- coding: utf-8 -*-
"""test_matrix.py — pairwise centered matrix self-consistency.

Checks, from results/extra/pairwise_centered_matrix.csv (15 pairs of
centered judge scores, R group):
  1. Additivity: for every triple (A, B, C) of models,
     |d(A,B) + d(B,C) - d(A,C)| <= 0.002 (point estimates; CIs are not
     required to be additive).
  2. Every 95% CI contains its own point estimate.
  3. Exactly 9 of 15 pairs are significant, and every significant pair is
     cross-tier (top tier: GLM-5.3-Flash, GLM-5.3, Kimi-K3; bottom tier:
     MiniMax-M3, Deepseek-V4.1-Flash, Hy3); no within-tier pair significant.
"""
from __future__ import annotations

import csv
import itertools
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
CSV = PKG / "results" / "extra" / "pairwise_centered_matrix.csv"

TOP = {"GLM-5.3-Flash", "GLM-5.3", "Kimi-K3"}
BOTTOM = {"MiniMax-M3", "Deepseek-V4.1-Flash", "Hy3"}


def load_pairs() -> dict[tuple[str, str], tuple[float, float, float]]:
    d: dict[tuple[str, str], tuple[float, float, float]] = {}
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            a, b = row["model_a"], row["model_b"]
            m, lo, hi = float(row["mean_diff"]), float(row["ci_lo"]), float(row["ci_hi"])
            d[(a, b)] = (m, lo, hi)
            d[(b, a)] = (-m, -hi, -lo)
    return d


class TestMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = load_pairs()
        cls.models = sorted({a for a, _ in cls.pairs})
        # 6 models, 15 ordered-as-given pairs = 30 mirrored entries
        assert len(cls.models) == 6, f"expected 6 models, got {cls.models}"
        assert len(cls.pairs) == 30, f"expected 30 mirrored entries, got {len(cls.pairs)}"

    def test_additivity_all_triples(self):
        """|d(A,B) + d(B,C) - d(A,C)| <= 0.002 for all 20 triples."""
        worst = 0.0
        for a, b, c in itertools.combinations(self.models, 3):
            for x, y, z in ((a, b, c), (a, c, b), (b, a, c)):
                d_xy = self.pairs[(x, y)][0]
                d_yz = self.pairs[(y, z)][0]
                d_xz = self.pairs[(x, z)][0]
                dev = abs(d_xy + d_yz - d_xz)
                worst = max(worst, dev)
                self.assertLessEqual(dev, 0.002,
                                     f"additivity violated for ({x},{y},{z}): {dev}")
        self.assertLessEqual(worst, 0.002)

    def test_ci_contains_point_estimate(self):
        for (a, b), (m, lo, hi) in self.pairs.items():
            self.assertLessEqual(lo, m + 1e-12, f"CI low > mean for {a}-{b}")
            self.assertGreaterEqual(hi, m - 1e-12, f"CI high < mean for {a}-{b}")

    def test_two_tier_structure(self):
        sig = [(a, b) for (a, b), (_, lo, hi) in self.pairs.items()
               if lo > 0 or hi < 0]
        # each significant ordered pair appears twice (mirror) -> 9 pairs = 18
        self.assertEqual(len(sig), 18, f"expected 9 significant pairs, got {len(sig)//2}")
        for a, b in sig:
            same_tier = (a in TOP and b in TOP) or (a in BOTTOM and b in BOTTOM)
            self.assertFalse(same_tier, f"within-tier pair significant: {a} vs {b}")
            cross = (a in TOP and b in BOTTOM) or (a in BOTTOM and b in TOP)
            self.assertTrue(cross, f"unexpected pair: {a} vs {b}")
        # the report's headline pairs
        m, lo, hi = self.pairs[("GLM-5.3", "Kimi-K3")]
        self.assertAlmostEqual(m, 0.015, places=3)
        m, lo, hi = self.pairs[("Deepseek-V4.1-Flash", "Hy3")]
        self.assertAlmostEqual(m, 0.020, places=3)
        self.assertAlmostEqual(lo, -0.163, places=3)
        self.assertAlmostEqual(hi, 0.183, places=3)


class TestMatrixRecompute(unittest.TestCase):
    """Regenerate the 15 pairs from the saved judge outputs (same seed) and
    compare with the shipped CSV; also verify the matrix uses the same centered
    scores as the main table (judge_scores_centered.csv, column R_adj)."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(PKG))
        from harness.analyze import centered_scores, pairwise_centered_matrix
        cls.cent = centered_scores(PKG / "results")
        cls.rows = pairwise_centered_matrix(PKG / "results", 10_000, 42)
        with open(PKG / "results" / "extra" / "judge_scores_centered.csv",
                  encoding="utf-8-sig", newline="") as f:
            cls.main = {r["model"]: r for r in csv.DictReader(f)}

    def test_recompute_matches_csv(self):
        with open(CSV, encoding="utf-8-sig", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(self.rows) - 1, 15)
        self.assertEqual(len(csv_rows), 15)
        for (a, b, m, lo, hi, sig), cr in zip(self.rows[1:], csv_rows):
            self.assertEqual((a, b), (cr["model_a"], cr["model_b"]))
            self.assertAlmostEqual(m, float(cr["mean_diff"]), delta=0.0005, msg=f"{a}-{b}")
            self.assertAlmostEqual(lo, float(cr["ci_lo"]), delta=0.0005, msg=f"{a}-{b} lo")
            self.assertAlmostEqual(hi, float(cr["ci_hi"]), delta=0.0005, msg=f"{a}-{b} hi")
            self.assertEqual(sig, cr["significant"], f"{a}-{b} sig")

    def test_same_centered_scores_as_main_table(self):
        for m, per_q in self.cent.items():
            adj = sum(per_q.values()) / len(per_q)
            self.assertAlmostEqual(adj, float(self.main[m]["R_adj"]), delta=0.0005,
                                   msg=f"{m} centered mean vs R_adj")


if __name__ == "__main__":
    unittest.main()
