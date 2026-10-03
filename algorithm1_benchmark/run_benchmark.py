"""Run the published Algorithm 1 synthetic 1,000-query benchmark."""
from pathlib import Path

from generate_synthetic_1000 import evaluate

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    evaluate(ROOT / "results/synthetic_1000")
