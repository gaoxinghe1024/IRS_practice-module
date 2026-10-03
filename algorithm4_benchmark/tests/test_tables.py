# -*- coding: utf-8 -*-
"""test_tables.py — scored.csv, tables 1-2 and the derived CSVs vs the saved data.

Offline, no model calls. Checks:
  1. results/scored.csv: every rule column equals a fresh harness/score.py pass
     over results/raw.jsonl.gz; every judge column equals a fresh aggregation of
     results/judge_raw.jsonl.gz (tolerance 1e-9).
  2. Tables 1-2 (CSV): means calculated here straight from results/scored.csv
     (not through harness/analyze.py) and 95% CIs from the same recipe (percentile
     bootstrap over questions, 10,000 resamples, random.Random(42) per cell) agree
     with the shipped files (tolerance 1e-9); the MD tables carry the same cells.
  3. `python harness/analyze.py --tables` regenerates tables 1-2 and the two
     derived CSVs from scored.csv into a scratch directory with the same content
     as the shipped files; the derived CSVs also satisfy their own invariants.
  4. Table 2 trap rows: per-model R-N equals the final / all row of
     results/extra/frank_rate_by_model.csv (tolerance 0.0005; that CSV has three
     decimals).
  5. Every CSV under results/ parses with the csv module, each row as wide as its
     header, no blank lines.
"""
from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "harness"))

import analyze as A  # noqa: E402
import judge as J  # noqa: E402
import score as S  # noqa: E402

RESULTS = PKG / "results"
MODELS = ["Hy3", "Deepseek-V4.1-Flash", "GLM-5.3", "GLM-5.3-Flash",
          "MiniMax-M3", "Kimi-K3"]
TOL = 1e-9

# Table 1: metric key -> (question type or None = all 80, scored.csv column, n)
TABLE1 = {
    "scene_top1": ("scene", "top1", 24),
    "episode_ok": ("scene", "episode_ok", 22),
    "all_satisfy": ("constraint", "all_satisfy", 16),
    "inkb_rate": (None, "in_kb_reply_ok", 80),
    "judge_avg": (None, "judge_avg", 80),
    "judge_all_avg": (None, "judge_all_avg", 80),
}
# Table 2: (question type, metric key, scored.csv column or "per_work", n)
TABLE2 = [
    ("scene", "top1", "top1", 24), ("scene", "episode_ok", "episode_ok", 22),
    ("constraint", "all_satisfy", "all_satisfy", 16),
    ("constraint", "per_work_rate", "per_work", 16),
    ("similar", "inkb_rate", "in_kb_reply_ok", 10),
    ("mood", "mood_ok", "mood_ok", 10),
    ("fact", "fact_ok", "fact_ok", 8),
    ("trap", "rule_frank", "rule_frank", 11),
]
GROUP_MD = {"R": "R (RAG)", "N": "N (LLM only)", "T": "T (template)"}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, strict=True))


def read_csv_raw(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, strict=True))


def flt(x: str):
    if x in ("True", "False"):
        return 1.0 if x == "True" else 0.0
    return float(x) if x != "" else None


def cell_values(rows: list[dict], label: str, group: str, qtype, col: str) -> list[float]:
    """One value per question for a (model, group) cell, from scored.csv rows."""
    vals = []
    for r in rows:
        if r["model_label"] != label or r["group"] != group:
            continue
        if qtype and r["type"] != qtype:
            continue
        if col == "rule_frank" and r["qid"] == "T10":
            continue
        if col == "per_work":
            n_rec = int(r["n_rec"])
            vals.append(int(r["n_rec_satisfy"]) / n_rec if n_rec > 0 else 0.0)
            continue
        v = flt(r[col])
        if v is not None:
            vals.append(v)
    return vals


def stats(vals: list[float]):
    lo, hi = A.boot_ci(vals, 10000, random.Random(42))
    return sum(vals) / len(vals), lo, hi


def fmt(m: float, lo: float, hi: float) -> str:
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def md_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[2:]  # drop header and separator


# --------------------------------------------------------------- 1. scored.csv

class TestScoredCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = S.load_jsonl(RESULTS / "raw.jsonl.gz")
        cls.scored = A.read_csv(RESULTS / "scored.csv")

    def test_header_is_the_documented_column_list(self):
        header = list(self.scored[0].keys())
        self.assertEqual(header, A.SCORED_COLS)
        self.assertNotIn("channel", header)

    def test_rule_columns_match_rescoring(self):
        testset = {q["id"]: q for q in S.load_jsonl(PKG / "data" / "testset_v1.jsonl")}
        aliases = S.load_aliases(PKG / "data" / "aliases.json")
        anime = S.load_anime(PKG / "data" / "kb" / "anime.db")
        fresh = S.score_records(self.raw, testset, aliases, anime, {})
        self.assertEqual(len(fresh), len(self.scored))
        rule_cols = [c for c in A.SCORED_COLS if c not in A.JUDGE_COLS]
        for new, old in zip(fresh, self.scored):
            for c in rule_cols:
                v = new.get(c)
                self.assertEqual("" if v is None else str(v), old[c],
                                 f"{old['qid']} {old['model_label']} {old['group']} {c}")

    def test_judge_columns_match_aggregation(self):
        jmap = {r["call_id"]: r for r in J.aggregate(RESULTS)}
        for src, old in zip(self.raw, self.scored):
            j = jmap.get(src["call_id"])
            for c in A.JUDGE_COLS:
                v = None if j is None else j.get(c)
                where = f"{old['qid']} {old['model_label']} {old['group']} {c}"
                if v is None or v == "":
                    self.assertEqual(old[c], "", where)
                elif isinstance(v, bool):
                    self.assertEqual(old[c], str(v), where)
                else:
                    self.assertAlmostEqual(float(old[c]), float(v), delta=TOL, msg=where)


# --------------------------------------------------- 2. tables 1-2 cell by cell

class TestTable1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored = A.read_csv(RESULTS / "scored.csv")
        cls.csv = read_csv_rows(RESULTS / "tables" / "table1_main_results.csv")

    def test_rows_and_columns(self):
        order = [("T (system default)", "T")] + [(m, g) for m in MODELS for g in "RN"]
        self.assertEqual([(r["model"], r["group"]) for r in self.csv], order)
        cols = ["model", "group"]
        for key in TABLE1:
            cols += [f"{key}_mean", f"{key}_ci_lo", f"{key}_ci_hi"]
        self.assertEqual(list(self.csv[0].keys()), cols)

    def test_cells_match_scored_csv(self):
        for row in self.csv:
            label = "T(system)" if row["group"] == "T" else row["model"]
            for key, (qtype, col, n) in TABLE1.items():
                vals = cell_values(self.scored, label, row["group"], qtype, col)
                where = f"{row['model']} {row['group']} {key}"
                self.assertEqual(len(vals), n, where)
                m, lo, hi = stats(vals)
                self.assertAlmostEqual(float(row[f"{key}_mean"]), m, delta=TOL, msg=where)
                self.assertAlmostEqual(float(row[f"{key}_ci_lo"]), lo, delta=TOL, msg=where)
                self.assertAlmostEqual(float(row[f"{key}_ci_hi"]), hi, delta=TOL, msg=where)

    def test_markdown_matches_csv(self):
        md = md_rows(RESULTS / "tables" / "table1_main_results.md")
        self.assertEqual(len(md), len(self.csv))
        for cells, row in zip(md, self.csv):
            self.assertEqual(cells[:2], [row["model"], GROUP_MD[row["group"]]])
            want = [fmt(float(row[f"{k}_mean"]), float(row[f"{k}_ci_lo"]),
                        float(row[f"{k}_ci_hi"])) for k in TABLE1]
            self.assertEqual(cells[2:], want, row["model"])


class TestTable2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored = A.read_csv(RESULTS / "scored.csv")
        cls.csv = read_csv_rows(RESULTS / "tables" / "table2_by_type.csv")

    def test_rows_and_columns(self):
        order = [(m, g, q, k) for m in MODELS for g in "RN" for q, k, _c, _n in TABLE2]
        self.assertEqual([(r["model"], r["group"], r["qtype"], r["metric"])
                          for r in self.csv], order)
        self.assertEqual(list(self.csv[0].keys()),
                         ["model", "group", "qtype", "n", "metric", "mean",
                          "ci_lo", "ci_hi"])

    def test_cells_match_scored_csv(self):
        spec = {(q, k): (c, n) for q, k, c, n in TABLE2}
        for row in self.csv:
            col, n = spec[(row["qtype"], row["metric"])]
            vals = cell_values(self.scored, row["model"], row["group"], row["qtype"], col)
            where = f"{row['model']} {row['group']} {row['qtype']} {row['metric']}"
            self.assertEqual(len(vals), n, where)
            self.assertEqual(int(row["n"]), n, where)
            m, lo, hi = stats(vals)
            self.assertAlmostEqual(float(row["mean"]), m, delta=TOL, msg=where)
            self.assertAlmostEqual(float(row["ci_lo"]), lo, delta=TOL, msg=where)
            self.assertAlmostEqual(float(row["ci_hi"]), hi, delta=TOL, msg=where)

    def test_markdown_matches_csv(self):
        md = md_rows(RESULTS / "tables" / "table2_by_type.md")
        self.assertEqual(len(md), len(self.csv))
        for cells, row in zip(md, self.csv):
            self.assertEqual(cells[:3], [row["model"], GROUP_MD[row["group"]], row["qtype"]])
            self.assertEqual(cells[4], row["n"])
            self.assertEqual(cells[5], fmt(float(row["mean"]), float(row["ci_lo"]),
                                           float(row["ci_hi"])))

    def test_frank_rate_csv_n_columns(self):
        rows = read_csv_rows(RESULTS / "extra" / "frank_rate_by_model.csv")
        self.assertEqual(list(rows[0].keys())[:4],
                         ["caliber", "slice", "n_per_model", "n_pooled"])
        per_slice = {"all": 11, "ex_real": 6, "only_real": 5}
        for r in rows:
            self.assertEqual(int(r["n_per_model"]), per_slice[r["slice"]], r["slice"])
            self.assertEqual(int(r["n_pooled"]), len(MODELS) * int(r["n_per_model"]))

    def test_trap_rn_matches_frank_rate_csv(self):
        frank = [r for r in read_csv_rows(RESULTS / "extra" / "frank_rate_by_model.csv")
                 if r["caliber"] == "final" and r["slice"] == "all"]
        self.assertEqual(len(frank), 1)
        mean = {(r["model"], r["group"]): float(r["mean"])
                for r in self.csv if r["metric"] == "rule_frank"}
        self.assertEqual(len(mean), 12)
        for m in MODELS:
            self.assertAlmostEqual(mean[(m, "R")] - mean[(m, "N")], float(frank[0][m]),
                                   delta=0.0005, msg=m)
        for r in self.csv:
            if r["metric"] == "rule_frank":
                self.assertEqual(r["n"], "11")  # T10 offtopic excluded


# ------------------------------------------- 3. regeneration + derived CSV checks

class TestRegeneration(unittest.TestCase):
    FILES = ["tables/table1_main_results.csv", "tables/table1_main_results.md",
             "tables/table2_by_type.csv", "tables/table2_by_type.md",
             "extra/inkb_reply_rate_per_reply.csv", "extra/out_of_kb_per_reply.csv"]

    def test_cli_regenerates_shipped_files(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "results"
            res.mkdir()
            for name in ("scored.csv", "raw.jsonl.gz"):
                shutil.copy(RESULTS / name, res / name)
            p = subprocess.run([sys.executable, "harness/analyze.py", "--tables",
                                "--results-dir", str(res)], cwd=PKG,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("TABLES_OK", p.stdout)
            for rel in self.FILES:
                if rel.endswith(".csv"):
                    self.assertEqual(read_csv_raw(res / rel), read_csv_raw(RESULTS / rel), rel)
                else:
                    self.assertEqual((res / rel).read_text(encoding="utf-8"),
                                     (RESULTS / rel).read_text(encoding="utf-8"), rel)
            self.assertFalse((res / "replay").exists())


    def test_cli_rescore_write_scored_is_byte_identical(self):
        # `--rescore --write-scored` rewrites scored.csv from raw.jsonl.gz and
        # judge_raw.jsonl.gz byte for byte
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "results"
            for rel in ("raw.jsonl.gz", "judge_raw.jsonl.gz", "retrieval.jsonl",
                        "extra/judge_label_maps.jsonl.gz"):
                (res / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(RESULTS / rel, res / rel)
            p = subprocess.run([sys.executable, "harness/analyze.py", "--rescore",
                                "--write-scored", "--fast", "--results-dir", str(res),
                                "--out-dir", str(Path(d) / "replay")], cwd=PKG,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual((res / "scored.csv").read_bytes(),
                             (RESULTS / "scored.csv").read_bytes())


class TestDerivedCsvs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored = A.read_csv(RESULTS / "scored.csv")

    def test_inkb_rate_per_model(self):
        rows = read_csv_rows(RESULTS / "extra" / "inkb_reply_rate_per_reply.csv")
        self.assertEqual(list(rows[0].keys()), ["model", "N_inkb_reply", "R_inkb_reply", "R-N"])
        self.assertEqual(sorted(r["model"] for r in rows), sorted(MODELS))
        for r in rows:
            rate = {}
            for g in "NR":
                vals = cell_values(self.scored, r["model"], g, None, "in_kb_reply_ok")
                self.assertEqual(len(vals), 80)
                rate[g] = sum(vals) / len(vals)
            self.assertAlmostEqual(float(r["N_inkb_reply"]), rate["N"], delta=TOL)
            self.assertAlmostEqual(float(r["R_inkb_reply"]), rate["R"], delta=TOL)
            self.assertAlmostEqual(float(r["R-N"]), rate["R"] - rate["N"], delta=TOL)

    def test_out_of_kb_per_reply(self):
        rows = read_csv_rows(RESULTS / "extra" / "out_of_kb_per_reply.csv")
        self.assertEqual(list(rows[0].keys()),
                         ["qid", "type", "model", "group", "n_out_kb", "n_echo",
                          "n_kb_variant", "n_out_kb_final", "echo_names", "kbvar_names",
                          "kept_names", "in_kb_reply_ok"])
        self.assertEqual(len(rows), len(self.scored))
        for r, s in zip(rows, self.scored):
            where = f"{s['qid']} {s['model_label']} {s['group']}"
            self.assertEqual((r["qid"], r["type"], r["model"], r["group"]),
                             (s["qid"], s["type"], s["model_label"], s["group"]), where)
            self.assertEqual(r["n_out_kb"], s["n_out_kb"], where)
            self.assertEqual(r["in_kb_reply_ok"], s["in_kb_reply_ok"], where)
            n_out, n_echo, n_var, n_final = (int(r[k]) for k in (
                "n_out_kb", "n_echo", "n_kb_variant", "n_out_kb_final"))
            self.assertEqual(n_echo + n_var + n_final, n_out, where)
            # final caliber: in-KB iff nothing is kept and some title is recognized
            self.assertEqual(r["in_kb_reply_ok"] == "True",
                             n_final == 0 and int(s["n_kb_hits"]) + n_out > 0, where)


# ------------------------------------------------------------ 5. strict CSV parse

class TestAllCsvFilesStrict(unittest.TestCase):
    def test_every_results_csv_parses_cleanly(self):
        bad = []
        files = sorted(RESULTS.rglob("*.csv"))
        self.assertGreater(len(files), 30)
        for p in files:
            rel = p.relative_to(RESULTS).as_posix()
            try:
                rows = read_csv_raw(p)
            except csv.Error as e:
                bad.append(f"{rel}: {e}")
                continue
            width = len(rows[0])
            for i, r in enumerate(rows[1:], 2):
                if not r:
                    bad.append(f"{rel}:L{i}: blank line")
                elif len(r) != width:
                    bad.append(f"{rel}:L{i}: {len(r)} columns, header has {width}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
