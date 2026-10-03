#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""presubmit_check.py — pre-submission gate for the packaged docs.

Standalone (no unittest runner). Scans README.md, REPORT.md, VALIDATION.md and
CHANGELOG.md for:
  1. "[TO-FILL" markers — must be resolved (or consciously accepted) before
     public submission;
  2. the process-narrative word list and old-value residues in README /
     REPORT / VALIDATION (CHANGELOG is exempt from the word list by contract;
     it is only scanned for TO-FILL markers).

Exit codes: 0 = clean, 1 = at least one hit (paths and lines printed).
Run:  python tools/presubmit_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

DOCS_WORDLIST = ("README.md", "REPORT.md", "VALIDATION.md")
DOCS_TOFILL = ("README.md", "REPORT.md", "VALIDATION.md", "CHANGELOG.md")

WORD_RE = re.compile(
    r"\b(earlier|older|previous|was wrong|corrected|bug|recomput\w*|flawed)\b"
    r"|before echo|\bround-\d|(?!\.)\bv[123]\b(?!\.\d)", re.I)

OLD_VALUES = (
    "+0.389", "+0.167, +0.611", "10/24", "10 of the 12",
    "\u22120.038", "\u22120.058", "-0.038", "-0.058",
    "sensitivity_scene_disputed_4questions", "pairwise_significance_centered",
    "\u5f85\u586b",
)

# Package-wide process-term list (mirrors tests/test_docs.py:TestPackageWordList).
# Scanned over every text file and every .gz decompressed; model replies are
# part of the scanned raw data (currently 0 hits — any future hit must be
# triaged manually and recorded in the audit).
PACKAGE_WORD_RE = re.compile(
    r"Round\s?\d|\u9879\s?\d|not ship" + "ped|github_" + "package"
    r"|\[REDACTED\]|e-A " + "IRS|cre" + "dit|mult" + "iplier"
    r"|\u79ef\u5206|\u500d\u7387|round-\d|this-" + "round", re.I)
TEXT_SUFFIXES = {".md", ".csv", ".json", ".py", ".txt", ".yaml", ".yml", ".svg"}


def package_text_hits() -> list[str]:
    import gzip
    hits: list[str] = []
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PKG).as_posix()
        if p.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                m = PACKAGE_WORD_RE.search(line)
                if m:
                    hits.append(f"[pkg-term] {rel}:L{i}: {m.group(0)}: {line.strip()[:70]}")
        elif p.suffix.lower() == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    m = PACKAGE_WORD_RE.search(line)
                    if m:
                        hits.append(f"[pkg-term] {rel}:L{i}: {m.group(0)}: {line.strip()[:70]}")
    return hits


def main() -> int:
    hits: list[str] = []
    for name in DOCS_TOFILL:
        text = (PKG / name).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "[TO-FILL" in line:
                hits.append(f"[TO-FILL] {name}:L{i}: {line.strip()[:90]}")
    for name in DOCS_WORDLIST:
        text = (PKG / name).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if WORD_RE.search(line):
                hits.append(f"[wordlist] {name}:L{i}: {line.strip()[:90]}")
            for old in OLD_VALUES:
                if old in line:
                    hits.append(f"[old-value {old!r}] {name}:L{i}: {line.strip()[:90]}")
    hits.extend(package_text_hits())

    if hits:
        print(f"presubmit_check: {len(hits)} hit(s)")
        for h in hits:
            print("  " + h)
        return 1
    print("presubmit_check: clean (no TO-FILL markers, no word-list hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
