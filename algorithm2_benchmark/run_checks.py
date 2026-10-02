"""Run the algorithm 2 checks and save their actual output to results/tests.txt (standard library only).

Covers the experiment's own tests (algorithm2_benchmark/tests: metric definitions, the fixture pipeline with
leakage, training, offline / service agreement and frozen-run reproducibility) and the website-side
algorithm 2 tests of the main suite (rules, strict service, bundle fallback, NumPy tree reader, API contract).

    .venv/bin/python algorithm2_benchmark/run_checks.py
"""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TESTS = ["algorithm2_benchmark/tests", "tests/unit/test_collection_algorithm2.py",
         "tests/integration/test_collection_algorithm2.py"]

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "pytest", "-vv", "-rs", *TESTS]   # -vv: the project adds -q to pytest's options
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"})
    text = (f"UTC: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"Command: python -m pytest -vv -rs {' '.join(TESTS)}\n\n" + result.stdout + result.stderr
            + f"\nExit code: {result.returncode}\n")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "tests.txt").write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(result.returncode)
