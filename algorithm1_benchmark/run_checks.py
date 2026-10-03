"""Run real unit/artifact checks and save their actual output for the handoff."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from kbbench.common import ROOT, utc_now

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic_1000",
                        help="Directory for the test log; checks read the published artifact set")
    args = parser.parse_args()
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    text = f"UTC: {utc_now()}\nCommand: python -m unittest discover -s tests -v\n\n" + result.stdout + result.stderr
    text += f"\nExit code: {result.returncode}\n"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.txt").write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(result.returncode)
