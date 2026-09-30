"""Replay delivered models/data without network access or the application system."""
from kbbench.evaluate import evaluate
from kbbench.report import report

if __name__ == "__main__":
    evaluate()
    report()
