"""Checks that the numbers quoted in README.md and REPORT.md match the CSV files."""
from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

MODEL_ALIAS = {
    "Deepseek": "Deepseek-V4.1-Flash",
    "MiniMax": "MiniMax-M3",
    "Kimi": "Kimi-K3",
}


def norm_model(name: str) -> str:
    name = name.strip()
    return MODEL_ALIAS.get(name, name)


def read(pkg_rel: str) -> str:
    return (PKG / pkg_rel).read_text(encoding="utf-8")


def fnum(s: str) -> float:
    s = s.strip().replace("\u2212", "-").replace(",", "")
    return float(s)


def load_matrix() -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    with open(PKG / "results/extra/pairwise_centered_matrix.csv",
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (norm_model(row["model_a"]), norm_model(row["model_b"]))
            out[key] = {
                "diff": fnum(row["mean_diff"]),
                "lo": fnum(row["ci_lo"]),
                "hi": fnum(row["ci_hi"]),
                "sig": row["significant"].strip().lower() == "yes",
            }
    return out


MATRIX = load_matrix()


def lookup(a: str, b: str):
    """Return (stats, flipped) for pair a-b, trying both stored directions."""
    ka, kb = (norm_model(a), norm_model(b))
    if (ka, kb) in MATRIX:
        return MATRIX[(ka, kb)], False
    if (kb, ka) in MATRIX:
        return MATRIX[(kb, ka)], True
    return None, False


REPORT = read("REPORT.md")
README = read("README.md")

# Appendix D table rows: | A | B | ±0.xxx | [lo, hi] | yes/no |
ROW_RE = re.compile(
    r"^\|\s*([A-Za-z0-9.\-]+)\s*\|\s*([A-Za-z0-9.\-]+)\s*\|\s*([+\u2212-]?0\.\d+)\s*"
    r"\|\s*\[([+\u2212-]?0\.\d+),\s*([+\u2212-]?0\.\d+)\]\s*\|\s*(yes|no)\s*\|",
    re.M)

# In-text citations: "X−Y = ±0.xxx" with optional "[lo, hi]" (X−Y with X the minuend)
CITE_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9.\-]*?)\s*[\u2212-]\s*([A-Za-z][A-Za-z0-9.\-]*?)\s*=\s*"
    r"([+\u2212-]?0\.\d+)(?:\s*\[([+\u2212-]?0\.\d+),\s*([+\u2212-]?0\.\d+)\])?")


class TestAppendixD(unittest.TestCase):
    def test_appendix_d_rows_match_csv(self):
        rows = ROW_RE.findall(REPORT)
        self.assertEqual(len(rows), 15, f"expected 15 appendix D rows, got {len(rows)}")
        for a, b, diff, lo, hi, sig in rows:
            m, flipped = lookup(a, b)
            self.assertIsNotNone(m, f"pair {a}-{b} not in CSV")
            s = -1.0 if flipped else 1.0
            self.assertAlmostEqual(fnum(diff), s * m["diff"], delta=0.005,
                                   msg=f"{a}-{b} point")
            self.assertAlmostEqual(fnum(lo), min(s * m["lo"], s * m["hi"]),
                                   delta=0.005, msg=f"{a}-{b} lo")
            self.assertAlmostEqual(fnum(hi), max(s * m["lo"], s * m["hi"]),
                                   delta=0.005, msg=f"{a}-{b} hi")
            self.assertEqual(sig == "yes", m["sig"], msg=f"{a}-{b} significance")

    def test_in_text_citations_match_csv(self):
        checked = 0
        for text, doc in ((REPORT, "REPORT.md"), (README, "README.md")):
            # skip appendix D table rows (covered above); citations are inline
            for a, b, diff, lo, hi in CITE_RE.findall(text):
                m, flipped = lookup(a, b)
                if m is None:
                    continue  # not a pairwise citation (e.g. rag lift ranges)
                s = -1.0 if flipped else 1.0
                self.assertAlmostEqual(fnum(diff), s * m["diff"], delta=0.005,
                                       msg=f"{doc} {a}-{b} point")
                if lo:
                    self.assertAlmostEqual(fnum(lo), min(s * m["lo"], s * m["hi"]),
                                           delta=0.005, msg=f"{doc} {a}-{b} lo")
                    self.assertAlmostEqual(fnum(hi), max(s * m["lo"], s * m["hi"]),
                                           delta=0.005, msg=f"{doc} {a}-{b} hi")
                checked += 1
        self.assertGreaterEqual(checked, 6,
                                "too few in-text pairwise citations parsed")


def load_named_recall() -> list[dict]:
    with open(PKG / "results/extra/named_recall_per_question.csv",
              encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def appendix_e_rows() -> list[list[str]]:
    block = REPORT.split("## Appendix E", 1)[1]
    rows = []
    for line in block.splitlines():
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0] in {"qid"} or (cells and cells[0] == "qid"):
                continue
            rows.append(cells)
    return rows


class TestAppendixEAndCounts(unittest.TestCase):
    def setUp(self):
        self.data = load_named_recall()
        self.by_qid = {r["qid"]: r for r in self.data}
        self.named = [r for r in self.data if r["in_named_set"] == "True"]
        self.assertEqual(len(self.named), 18)
        self.assertEqual(sum(r["type"] == "fact" for r in self.named), 8)
        self.assertEqual(sum(r["type"] == "similar" for r in self.named), 10)
        # the 6 scene rows are excluded from the named set
        for r in self.data:
            if r["type"] == "scene":
                self.assertEqual(r["in_named_set"], "False", r["qid"])
                self.assertIn(r["named_by"], ("generic_alias", "character_alias"), r["qid"])

    def test_appendix_e_rows_match_csv(self):
        rows = appendix_e_rows()
        self.assertEqual(len(rows), 18, f"expected 18 appendix E rows, got {len(rows)}")
        for cells in rows:
            qid, qtype, work, named_by, score, pos, in5, stage, wrow, sline, anyp, method = cells
            r = self.by_qid[qid]
            self.assertEqual(r["in_named_set"], "True", qid)
            self.assertEqual(qtype, r["type"], qid)
            self.assertEqual(work, r["title"], qid)
            self.assertEqual(named_by, r["named_by"], qid)
            self.assertAlmostEqual(float(score), float(r["raw_score"]), delta=1e-6, msg=qid)
            self.assertEqual(pos, r["post_demotion_pos"] or "-", qid)
            self.assertEqual(in5, r["in_top5"], qid)
            self.assertEqual(stage, r["exit_stage"], qid)
            self.assertEqual(wrow, r["own_work_row"], qid)
            self.assertEqual(sline, r["scene_line_hit"], qid)
            self.assertEqual(anyp, r["in_prompt_any"], qid)
            self.assertEqual(method, r["hit_method"] or "-", qid)

    def test_subset_own_row_in_top5(self):
        for r in self.named:
            if r["own_work_row"] == "True":
                self.assertEqual(r["in_top5"], "True", r["qid"])

    def test_counts_computed_and_verbatim(self):
        named = self.named
        n_any = sum(r["in_prompt_any"] == "True" for r in named)
        n_row = sum(r["own_work_row"] == "True" for r in named)
        sline_only = sorted(r["qid"] for r in named
                            if r["in_prompt_any"] == "True"
                            and r["own_work_row"] == "False")
        absent = [r for r in named if r["in_prompt_any"] == "False"]
        n_absent = len(absent)
        n_abs_demoted = sum(r["exit_stage"] == "demoted_out_of_top5" for r in absent)
        n_abs_filtered = sum(r["exit_stage"] == "constraint_filtered" for r in absent)
        no_row = [r for r in named if r["own_work_row"] == "False"]
        n_no_row = len(no_row)
        n_no_row_demoted = sum(r["exit_stage"] == "demoted_out_of_top5" for r in no_row)
        n_no_row_filtered = sum(r["exit_stage"] == "constraint_filtered" for r in no_row)
        pres_fact = sum(r["type"] == "fact" and r["in_prompt_any"] == "True" for r in named)
        pres_sim = sum(r["type"] == "similar" and r["in_prompt_any"] == "True"
                       for r in named)
        abs_fact = sum(r["type"] == "fact" and r["in_prompt_any"] == "False" for r in named)
        abs_sim = sum(r["type"] == "similar" and r["in_prompt_any"] == "False"
                      for r in named)

        self.assertEqual((n_any, n_row, len(sline_only)), (7, 4, 3))
        self.assertEqual(sline_only, ["X01", "X02", "X05"])
        self.assertEqual((n_absent, n_abs_demoted, n_abs_filtered), (11, 10, 1))
        self.assertEqual((n_no_row, n_no_row_demoted, n_no_row_filtered), (14, 12, 2))
        self.assertEqual((pres_fact, pres_sim), (3, 4))
        self.assertEqual((abs_fact, abs_sim), (5, 6))

        # verbatim in REPORT.md (key finding / retrieval layer wording)
        phrase_groups = (
            ("4 as an own work-entry row", "4 with an own work-entry row"),
            ("3 only via a famous-scene line (X01, X02, X05)",
             "3 only via famous-scene lines"),
            ("10 pushed out of the top-5 candidates by mention-demotion",
             "10 of the 11 misses are pushed out of the top-5 candidates"),
            ("1 removed by the constraint filter",
             "1 similar question is removed by the constraint filter"),
            ("absent for 11 of 18", "absent from the R prompt's KB text for 11 of 18"),
            ("present for the other 7 (4 as an own work-entry row",
             "present for the other 7 of 18"),
        )
        for group in phrase_groups:
            self.assertTrue(any(p in REPORT for p in group),
                            f"none of {group} found in REPORT.md")


def load_false_absence() -> list[dict]:
    with open(PKG / "results/extra/false_absence_per_reply.csv",
              encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestFalseAbsenceCounts(unittest.TestCase):
    def test_fa_counts_match_csv_and_docs(self):
        rows = load_false_absence()
        self.assertEqual(len(rows), 216)  # 18 named questions x 6 models x 2 groups
        self.assertEqual({r["qid"] for r in rows},
                         {r["qid"] for r in load_named_recall()
                          if r["in_named_set"] == "True"})
        r_decl = [r for r in rows if r["group"] == "R"
                  and r["declares_absence"] == "True"]
        n_decl = [r for r in rows if r["group"] == "N"
                  and r["declares_absence"] == "True"]
        strict = [r for r in rows if r["strictly_false"] == "True"]
        self.assertEqual(len(r_decl), 43)
        self.assertEqual(sum(r["type"] == "fact" for r in r_decl), 27)
        self.assertEqual(sum(r["type"] == "similar" for r in r_decl), 16)
        self.assertEqual(sum(r["type"] == "scene" for r in r_decl), 0)
        self.assertEqual(len(n_decl), 0)
        self.assertEqual([(r["qid"], r["model"]) for r in strict],
                         [("X05", "Hy3")])
        self.assertIn("43 false-absence declarations in the R group", REPORT)


def load_scene_cluster() -> dict[tuple[str, str], tuple[float, float, float]]:
    out = {}
    with open(PKG / "results/extra/scene_sensitivity_cluster.csv",
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["slice"], row["metric"])] = (
                float(row["point"]), float(row["ci_lo"]), float(row["ci_hi"]))
    return out


class TestSceneSensitivity(unittest.TestCase):
    def test_report_and_readme_numbers_match_cluster_csv(self):
        cc = load_scene_cluster()
        for doc, text, slices in (
            ("REPORT.md", REPORT, ("all_22q", "excl_S08S12_18q")),
            ("README.md", README, ("excl_S08S12_18q",)),
        ):
            for slice_ in slices:
                vals = {"all_22q": ("+0.470", "+0.303", "+0.629"),
                        "excl_S08S12_18q": ("+0.426", "+0.241", "+0.611")}[slice_]
                point, lo, hi = vals
                self.assertIn(f"{point} [{lo}, {hi}]", text,
                              f"{doc} missing {slice_} interval")
                p, l, h = cc[(slice_, "R-N")]
                for got, want in ((point, p), (lo, l), (hi, h)):
                    self.assertAlmostEqual(float(got), round(want, 3), delta=5e-4,
                                           msg=f"{doc} {slice_}")
                if doc == "REPORT.md":
                    # point estimates of R and N rates appear rounded to 3 dp
                    r = cc[(slice_, "R")]
                    n = cc[(slice_, "N")]
                    self.assertIn(f"{r[0]:.3f}", text)
                    self.assertIn(f"{n[0]:.3f}", text)


def parse_model_table(text: str, header_hint: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header_hint in line:
            rows = []
            for l in lines[i + 2:]:
                if not l.strip().startswith("|"):
                    break
                cells = [c.strip() for c in l.strip("|").split("|")]
                rows.append(cells)
            return {r[0]: r[1:] for r in rows}
    raise AssertionError(f"model table not found: {header_hint}")


class TestModelTables(unittest.TestCase):
    def test_readme_and_report_tables_agree(self):
        rm = parse_model_table(REPORT, "R raw")
        rd = parse_model_table(README, "Judge (raw)")
        self.assertEqual(set(rm), set(rd), "same model rows")
        for model, rcells in rm.items():
            dcells = rd[model]
            # R raw | centered (rank) | >20 s | p95 | output tokens
            self.assertAlmostEqual(float(rcells[0]), float(dcells[0]), msg=model)
            self.assertEqual(rcells[1].replace(" ", ""), dcells[1].replace(" ", ""),
                             msg=f"{model} centered")
            self.assertEqual(rcells[2].split("(")[0].strip(),
                             dcells[2].strip(), msg=f"{model} >20s")
            self.assertAlmostEqual(
                float(rcells[3].replace("s", "").strip()),
                float(dcells[3].replace("s", "").strip()), msg=f"{model} p95")
            self.assertEqual(int(rcells[4]), int(dcells[4]), msg=f"{model} tokens")


PATH_RE = re.compile(r"`((?:algo|data|harness|tools|tests|results)/[^`]+)`")
PATH_ALLOWLIST = {"results/replay/"}


class TestBacktickPaths(unittest.TestCase):
    def test_referenced_paths_exist(self):
        missing = []
        for doc, text in (("README.md", README), ("REPORT.md", REPORT)):
            for p in PATH_RE.findall(text):
                p_clean = p.split()[0].split(":")[0].rstrip(".")
                if p_clean in PATH_ALLOWLIST or not p_clean:
                    continue
                if not (PKG / p_clean).exists():
                    missing.append(f"{doc}: {p_clean}")
        self.assertEqual(missing, [], "dangling package paths")


def read_csv_rows(rel: str) -> list[dict]:
    with open(PKG / rel, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, strict=True))


def inkb_ranges() -> tuple[str, str, str]:
    """(N range, R range, paired-lift range) over the six models, as quoted in the docs."""
    rows = read_csv_rows("results/extra/inkb_reply_rate_per_reply.csv")
    n = [float(r["N_inkb_reply"]) for r in rows]
    r_ = [float(r["R_inkb_reply"]) for r in rows]
    lift = [float(r["R-N"]) for r in rows]
    return (f"{min(n):.2f}\u2013{max(n):.2f}", f"{min(r_):.2f}\u2013{max(r_):.2f}",
            f"+{min(lift):.2f} to +{max(lift):.2f}")


class TestInKbDefinitionAndRanges(unittest.TestCase):
    def test_appendix_b_in_kb_row_states_both_conditions(self):
        rows = [l for l in REPORT.splitlines() if l.startswith("| In-KB citation rate")]
        self.assertEqual(len(rows), 1)
        for needle in ("echo titles", "in-KB spelling variants",
                       "no out-of-KB title remains", "at least one"):
            self.assertIn(needle, rows[0])
        self.assertEqual(sum(l.startswith("| Per-work constraint rate")
                             for l in REPORT.splitlines()), 1)

    def test_ranges_in_docs_match_csv(self):
        n_rng, r_rng, lift = inkb_ranges()
        self.assertIn(f"Reply-level in-KB citation rate: {r_rng} in R vs {n_rng} in N "
                      f"(paired lift {lift})", REPORT)
        self.assertIn(f"{n_rng} in N \u2192 **{r_rng} of replies in R**, "
                      f"paired lift **{lift}**", REPORT)
        self.assertIn(lift, README)

    def test_r_out_of_kb_reply_rate_matches_csv(self):
        per: dict[str, list[int]] = {}
        for r in read_csv_rows("results/extra/out_of_kb_per_reply.csv"):
            if r["group"] == "R":
                k = per.setdefault(r["model"], [0, 0])
                k[1] += 1
                k[0] += int(r["n_out_kb_final"]) > 0
        self.assertEqual(sum(v[1] for v in per.values()), 480)
        rates = [a / b for a, b in per.values()]
        self.assertIn(f"R's out-of-KB reply rate is {min(rates):.2f}\u2013{max(rates):.2f}",
                      REPORT)


class TestOutOfKbReview(unittest.TestCase):
    def test_review_rows_match_residual_replies(self):
        per = read_csv_rows("results/extra/out_of_kb_per_reply.csv")
        residual = [r for r in per if r["group"] == "R" and int(r["n_out_kb_final"]) > 0]
        review = read_csv_rows("results/extra/out_of_kb_review.csv")
        self.assertEqual(list(review[0].keys()),
                         ["qid", "model", "residual_title", "category", "disclosed",
                          "evidence"])
        self.assertEqual([(r["qid"], r["model"], r["residual_title"]) for r in review],
                         [(r["qid"], r["model"], r["kept_names"]) for r in residual])
        for r in review:
            self.assertTrue(r["category"], r["qid"])
            self.assertLessEqual(len(r["evidence"]), 40, r["qid"])
            self.assertIn(r["disclosed"], ("yes", "n/a"), r["qid"])  # never "no"
        n = len(residual)
        self.assertGreater(n, 0)
        # the docs quote the count and the "0 of 480" conclusion the review supports
        self.assertIn(f"the {n} R replies that keep a residual out-of-KB title", REPORT)
        self.assertIn(f"{n} of the 480 R replies keep such a residual", REPORT)
        self.assertIn("0 of 480 R replies recommend an out-of-KB title without disclosure",
                      REPORT)
        self.assertIn("`results/extra/out_of_kb_review.csv`", REPORT)
        self.assertIn("title extraction treats a quoted character name", REPORT.lower())
        self.assertIn("upper bound", REPORT)


if __name__ == "__main__":
    unittest.main()
