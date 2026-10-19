# -*- coding: utf-8 -*-
"""test_figures.py — fig1-fig4 are drawn from packaged data (tools/make_figures.py).

Offline, no model calls. Checks:
  1. The data behind each figure equals the packaged tables: fig1 / fig3 judge
     score and in-KB rate with CIs = table 1, fig2 rows = table 3 plus the
     reply-level in-KB paired lift (= inkb_reply_rate_per_reply.csv), fig3 x =
     table 4 "Avg Net Total" for each group, fig4 = intent_threshold_sweep.csv.
  2. With matplotlib installed (skipped otherwise): `python tools/make_figures.py`
     and `python harness/analyze.py --figures` both write fig1-fig4 as PNG + SVG,
     the SVGs are valid XML, identical between the two entry points, and carry no
     timestamps, local paths or process wording.
"""
from __future__ import annotations

import csv
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RESULTS = PKG / "results"
spec = importlib.util.spec_from_file_location("make_figures", PKG / "tools" / "make_figures.py")
MF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MF)

MODELS = MF.MODELS
TOL = 1e-9


def rows(rel: str) -> list[dict]:
    with open(RESULTS / rel, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestFigureData(unittest.TestCase):
    def test_fig1_matches_table1(self):
        d = MF.fig1_data(RESULTS)
        for r in rows("tables/table1_main_results.csv"):
            if r["group"] == "T":
                self.assertAlmostEqual(d["T"]["judge"], float(r["judge_avg_mean"]), delta=TOL)
                self.assertAlmostEqual(d["T"]["inkb"], float(r["inkb_rate_mean"]), delta=TOL)
                continue
            cell = d[r["group"]][r["model"]]
            for key, col in (("judge", "judge_avg"), ("inkb", "inkb_rate")):
                want = tuple(float(r[f"{col}_{s}"]) for s in ("mean", "ci_lo", "ci_hi"))
                for a, b in zip(cell[key], want):
                    self.assertAlmostEqual(a, b, delta=TOL, msg=f"{r['model']} {r['group']}")

    def test_inkb_values_are_reply_level(self):
        # fig1 right panel / fig2 in-KB row use in_kb_reply_ok: N 0.20-0.38,
        # R 0.94-1.00, paired lift +0.59 to +0.80 (inkb_reply_rate_per_reply.csv)
        d = MF.fig1_data(RESULTS)
        lifts = {r["model"]: float(r["R-N"])
                 for r in rows("extra/inkb_reply_rate_per_reply.csv")}
        fig2 = dict(MF.fig2_data(RESULTS))["In-KB citation rate (reply-level)"]
        for model, diff, lo, hi, sig in fig2:
            self.assertAlmostEqual(diff, lifts[model], delta=TOL, msg=model)
            self.assertAlmostEqual(diff, d["R"][model]["inkb"][0] - d["N"][model]["inkb"][0],
                                   delta=TOL, msg=model)
            self.assertLess(lo, diff)
            self.assertGreater(hi, diff)
            self.assertTrue(sig, model)  # CI excludes 0 for every model
        n = [d["N"][m]["inkb"][0] for m in MODELS]
        r_ = [d["R"][m]["inkb"][0] for m in MODELS]
        self.assertEqual(f"{min(n):.2f}–{max(n):.2f}", "0.20–0.38")
        self.assertEqual(f"{min(r_):.2f}–{max(r_):.2f}", "0.94–1.00")
        lift = [x[1] for x in fig2]
        self.assertEqual(f"+{min(lift):.2f} to +{max(lift):.2f}", "+0.59 to +0.80")

    def test_fig2_other_rows_match_table3(self):
        groups = MF.fig2_data(RESULTS)
        self.assertEqual([g for g, _ in groups],
                         ["Judge score (overall)", "In-KB citation rate (reply-level)",
                          "Episode OK (scene)", "All constraints (constraint)",
                          "Frank rate (trap)", "Fact OK (fact)"])
        metric = {"Judge score (overall)": "judge_avg", "Episode OK (scene)": "episode_ok",
                  "All constraints (constraint)": "all_satisfy", "Frank rate (trap)": "rule_frank",
                  "Fact OK (fact)": "fact_ok"}
        t3 = {(r["Model"], r["Metric"]): r for r in rows("tables/table3_rag_lift.csv")}
        for label, rs in groups:
            self.assertEqual([m for m, *_ in rs], MODELS)  # same model order as fig1
            if label not in metric:
                continue
            for model, diff, lo, hi, sig in rs:
                r = t3[(model, metric[label])]
                clo, chi = (float(x) for x in re.findall(r"-?\d+\.\d+", r["95% CI"]))
                self.assertEqual((diff, lo, hi, sig),
                                 (float(r["Mean Diff (R-N)"]), clo, chi,
                                  r["Significant"] == "yes"), f"{label} {model}")

    def test_fig3_x_is_each_groups_own_net_total(self):
        d = MF.fig3_data(RESULTS)
        t4 = {(r["Model"], r["Group"]): float(r["Avg Net Total"])
              for r in rows("tables/table4_token_usage.csv")}
        for g in ("N", "R"):
            for m in MODELS:
                self.assertEqual(d[g][m][0], t4[(m, g)], f"{m} {g}")
        for m in MODELS:
            self.assertNotEqual(d["N"][m][0], d["R"][m][0], m)
        want = {"N": [2075, 1587, 1465, 641, 610, 415], "R": [1229, 1058, 1060, 742, 734, 691]}
        for g in ("N", "R"):
            self.assertEqual(sorted(round(v[0]) for v in d[g].values()), sorted(want[g]))
        f1 = MF.fig1_data(RESULTS)
        for g in ("N", "R"):
            for m in MODELS:
                self.assertEqual(d[g][m][1:], f1[g][m]["judge"], f"{m} {g}")

    def test_fig4_matches_sweep_csv(self):
        data = MF.fig4_data(RESULTS)
        src = rows("extra/intent_threshold_sweep.csv")
        self.assertEqual(len(data), len(src))
        self.assertIn(0.34, [r["t"] for r in data])
        self.assertIn(0.35, [r["t"] for r in data])
        for d, r in zip(data, src):
            self.assertEqual((d["t"], d["easy"], d["hard"], d["misfire"]),
                             (float(r["t"]), float(r["easy_rate"]), float(r["hard_rate"]),
                              float(r["misfire_rate"])))


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
class TestRendering(unittest.TestCase):
    NAMES = ["fig1_grouped_bars", "fig2_rag_lift", "fig3_quality_vs_tokens",
             "fig4_threshold_sweep"]

    def test_both_entry_points_render_identical_valid_svgs(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "results"
            for rel in ("scored.csv", "tables/table1_main_results.csv",
                        "tables/table3_rag_lift.csv", "tables/table4_token_usage.csv",
                        "extra/intent_threshold_sweep.csv"):
                (res / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(RESULTS / rel, res / rel)
            p = subprocess.run([sys.executable, "harness/analyze.py", "--figures",
                                "--results-dir", str(res)], cwd=PKG,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("FIGURES_OK", p.stdout)
            q = subprocess.run([sys.executable, "tools/make_figures.py", "--results-dir",
                                str(res), "--out-dir", str(Path(d) / "again")], cwd=PKG,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(q.returncode, 0, q.stdout + q.stderr)
            for name in self.NAMES:
                png = (res / "figures" / f"{name}.png").read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n", name)
                svg = (res / "figures" / f"{name}.svg").read_bytes()
                self.assertEqual(svg, (Path(d) / "again" / f"{name}.svg").read_bytes(), name)
                ET.fromstring(svg)  # valid XML
                text = svg.decode("utf-8")
                self.assertNotIn("<dc:date>", text)
                self.assertIsNone(re.search(
                    r"round-\d|this-" + "round|(?<![A-Za-z0-9])[A-Za-z]:\\\\|/Users/|/home/",
                    text), name)
            self.assertFalse((res / "replay").exists())


if __name__ == "__main__":
    unittest.main()
