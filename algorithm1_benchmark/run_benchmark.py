"""Replay delivered models/data without network access or the application system."""
from kbbench.evaluate import evaluate
from kbbench.report import report
from kbbench.common import ROOT

if __name__ == "__main__":
    if (ROOT / "data/optimization_protocol.json").exists():
        evaluate(output=ROOT / "results/baseline", variant="baseline")
    evaluate()
    report()
