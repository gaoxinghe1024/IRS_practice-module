"""Package item 3(c): replay-path tests.

* both entry styles run (`python harness/analyze.py` and `python -m harness.analyze`)
* the default replay (over results/scored.csv) and --rescore (re-scoring the
  raw replies in results/raw.jsonl.gz via harness/score.py) produce identical
  replay summaries (tolerance 1e-9 on every shared cell)
* the replay summary reproduces the headline numbers quoted in REPORT.md:
  factoid R 19/48 vs N 48/48 (final fact scorer), aggregate frank lift +0.62
  (n = 66 paired trap replies, T10 excluded), and the reply-level in-KB
  citation ranges (N, R and the paired lift), which are calculated from
  results/scored.csv and compared with the strings in REPORT.md / README.md.
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def _run_analyze(module_style: bool, out_dir: Path) -> list[dict]:
    args = [sys.executable] + (["-m", "harness.analyze"] if module_style
                               else ["harness/analyze.py"])
    cmd = args + ["--fast", "--out-dir", str(out_dir)]
    p = subprocess.run(cmd, cwd=PKG, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"analyze failed ({'module' if module_style else 'script'} "
                             f"entry):\n{p.stdout}\n{p.stderr}")
    with open(Path(out_dir) / "replay_summary.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _key(row):
    return (row["model_label"], row["group"])


def _cells(rows):
    out = {}
    for r in rows:
        for k, v in r.items():
            out[(_key(r), k)] = v
    return out


def _load_scored():
    with open(PKG / "results" / "scored.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class TestBothEntriesRun(unittest.TestCase):
    def test_script_and_module_entry_agree(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = _run_analyze(False, Path(d1) / "script")
            b = _run_analyze(True, Path(d2) / "module")
        self.assertEqual(len(a), len(b))
        ca, cb = _cells(a), _cells(b)
        self.assertEqual(set(ca), set(cb))
        for k, va in ca.items():
            vb = cb[k]
            try:
                self.assertLessEqual(abs(float(va) - float(vb)), 1e-9, k)
            except ValueError:
                self.assertEqual(va, vb, k)


class TestPathsAgree(unittest.TestCase):
    def test_default_and_rescore_summaries_identical(self):
        # default replay reads scored.csv; --rescore re-scores raw.jsonl.gz.
        # Both must land on the same summary (final calibers everywhere).
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            default = _run_analyze(False, Path(d1) / "default")
            cmd = [sys.executable, "harness/analyze.py", "--rescore", "--fast",
                   "--out-dir", str(Path(d2) / "rescore")]
            p = subprocess.run(cmd, cwd=PKG, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            rescore = list(csv.DictReader(
                open(Path(d2) / "rescore" / "replay_summary.csv", encoding="utf-8-sig")))
        cd, cr = _cells(default), _cells(rescore)
        self.assertEqual(set(cd), set(cr))
        diffs = []
        for k, va in cd.items():
            vb = cr[k]
            try:
                if abs(float(va) - float(vb)) > 1e-9:
                    diffs.append((k, va, vb))
            except ValueError:
                if va != vb:
                    diffs.append((k, va, vb))
        self.assertEqual(diffs, [])


class TestSummaryMatchesReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _load_scored()
        cls.report = (PKG / "REPORT.md").read_text(encoding="utf-8")

    def test_fact_totals(self):
        # REPORT: final fact scorer, R 19/48 and N 48/48
        tot = defaultdict(lambda: [0, 0])
        for r in self.rows:
            if r["type"] == "fact" and r.get("fact_ok", "") != "":
                tot[r["group"]][1] += 1
                if r["fact_ok"] == "True":
                    tot[r["group"]][0] += 1
        self.assertEqual((tot["R"][0], tot["R"][1]), (19, 48))
        self.assertEqual((tot["N"][0], tot["N"][1]), (48, 48))
        self.assertIn("R 19/48", self.report)

    def test_frank_aggregate(self):
        # REPORT: fixed-regex frank, aggregate R-N +0.62 over n = 66 paired
        # trap replies (T10 offtopic excluded)
        cnt = defaultdict(lambda: [0, 0])
        for r in self.rows:
            if (r["type"] == "trap" and r["qid"] != "T10"
                    and r["group"] in ("R", "N") and r.get("rule_frank", "") != ""):
                cnt[r["group"]][1] += 1
                if r["rule_frank"] == "True":
                    cnt[r["group"]][0] += 1
        self.assertEqual(cnt["R"][1], 66)
        self.assertEqual(cnt["N"][1], 66)
        lift = cnt["R"][0] / 66 - cnt["N"][0] / 66
        self.assertLessEqual(abs(lift - 0.6212), 0.0005)
        self.assertIn("aggregate +0.62, n = 66", self.report)

    def test_inkb_reply_ranges(self):
        # REPORT / README: reply-level in-KB citation rate ranges over the six
        # models (N, R) and the paired lift (per-model R - N), all calculated here
        per = defaultdict(lambda: {"N": [0, 0], "R": [0, 0]})
        for r in self.rows:
            if r["group"] in ("N", "R") and r.get("in_kb_reply_ok", "") != "":
                cell = per[r["model_label"]][r["group"]]
                cell[1] += 1
                if r["in_kb_reply_ok"] == "True":
                    cell[0] += 1
        n_rates = [per[m]["N"][0] / per[m]["N"][1] for m in per]
        r_rates = [per[m]["R"][0] / per[m]["R"][1] for m in per]
        lifts = [per[m]["R"][0] / per[m]["R"][1] - per[m]["N"][0] / per[m]["N"][1]
                 for m in per]
        self.assertEqual(len(per), 6)
        n_rng = f"{min(n_rates):.2f}–{max(n_rates):.2f}"
        r_rng = f"{min(r_rates):.2f}–{max(r_rates):.2f}"
        lift_rng = f"+{min(lifts):.2f} to +{max(lifts):.2f}"
        for needle in (f"{n_rng} in N", f"{r_rng} of replies in R", lift_rng):
            self.assertIn(needle, self.report, needle)
        readme = (PKG / "README.md").read_text(encoding="utf-8")
        self.assertIn(lift_rng, readme)
        # per-model source table in the package matches the same numbers
        tbl = list(csv.DictReader(open(
            PKG / "results" / "extra" / "inkb_reply_rate_per_reply.csv",
            encoding="utf-8-sig")))
        self.assertEqual(len(tbl), 6)
        for row in tbl:
            m = row["model"]
            self.assertAlmostEqual(float(row["N_inkb_reply"]),
                                   per[m]["N"][0] / per[m]["N"][1], places=9)
            self.assertAlmostEqual(float(row["R_inkb_reply"]),
                                   per[m]["R"][0] / per[m]["R"][1], places=9)
            self.assertAlmostEqual(float(row["R-N"]),
                                   per[m]["R"][0] / per[m]["R"][1]
                                   - per[m]["N"][0] / per[m]["N"][1], places=9)

    def test_scoring_from_raw_reproduces_fact_and_frank(self):
        # the --rescore path (harness/score.py over raw.jsonl.gz) must land on
        # the same headline counts as the shipped scored.csv
        from harness import score as S
        testset = {q["id"]: q for q in S.load_jsonl(PKG / "data" / "testset_v1.jsonl")}
        aliases = S.load_aliases(PKG / "data" / "aliases.json")
        anime = S.load_anime(PKG / "data" / "kb" / "anime.db")
        raw = S.load_jsonl(PKG / "results" / "raw.jsonl.gz")
        scored = S.score_records(raw, testset, aliases, anime, {})
        fact = defaultdict(lambda: [0, 0])
        frank = defaultdict(lambda: [0, 0])
        for r in scored:
            if r["type"] == "fact" and "fact_ok" in r:
                fact[r["group"]][1] += 1
                fact[r["group"]][0] += int(r["fact_ok"])
            if (r["type"] == "trap" and r["qid"] != "T10"
                    and r["group"] in ("R", "N") and "rule_frank" in r):
                frank[r["group"]][1] += 1
                frank[r["group"]][0] += int(r["rule_frank"])
        self.assertEqual((fact["R"][0], fact["R"][1]), (19, 48))
        self.assertEqual((fact["N"][0], fact["N"][1]), (48, 48))
        lift = frank["R"][0] / 66 - frank["N"][0] / 66
        self.assertLessEqual(abs(lift - 0.6212), 0.0005)


if __name__ == "__main__":
    unittest.main()
