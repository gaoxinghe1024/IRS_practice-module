"""Command line for the algorithm-2 experiments.

    python -m algorithm2_benchmark <stage> --run <run_id> [--level fixture|smoke|main]

stages: audit, prepare, split, tune, train, freeze, evaluate-test, benchmark, verify-integration, report,
run-all (runs the stages in order, each in a fresh interpreter, skipping those already finished for this run).
A new run needs --level; an existing run keeps its stored config.json. After `freeze`, `tune` and
`train` refuse to run, and `evaluate-test` only reproduces the frozen evaluation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from algorithm2_benchmark.pipeline.common import (
    Logger,
    environment,
    git_state,
    now,
    peak_rss_mb,
    stage_done,
    write_json,
)
from algorithm2_benchmark.pipeline.config import open_run
from app.core.config import PROJECT_ROOT

ORDER = ["audit", "prepare", "split", "tune", "train", "freeze", "evaluate-test", "benchmark", "verify-integration",
         "report"]


def stage_audit(run, log):
    from algorithm2_benchmark.pipeline.audit import audit

    return audit(run, log)


def stage_prepare(run, log):
    from algorithm2_benchmark.pipeline.common import mark_stage
    from algorithm2_benchmark.pipeline.data import prepare

    out = prepare(run, log)
    mark_stage(run.root, "prepare", {"catalog_hash": out["catalogue"]["catalog_hash"]})
    return out


def stage_split(run, log):
    from algorithm2_benchmark.pipeline.common import mark_stage
    from algorithm2_benchmark.pipeline.split import split

    out = split(run, log)
    mark_stage(run.root, "split", {"queries": out["queries"]["rows"]})
    return out


def spawner(run, log, threads: int | None = None):
    """Run an isolated sub-stage in a fresh interpreter (LightGBM and PyTorch never share a process)."""
    def spawn(stage: str) -> None:
        cmd = [sys.executable, "-m", "algorithm2_benchmark", stage, "--run", run.run_id, "--root",
               str(run.root.parent)]
        if threads:
            cmd += ["--threads", str(threads)]
        log(f"[spawn] {stage}")
        r = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if r.returncode != 0:
            raise SystemExit(f"sub-stage {stage} failed with exit code {r.returncode}")
    return spawn


def stage_tune(run, log):
    from algorithm2_benchmark.pipeline.train import tune

    return tune(run, log, spawner(run, log))


def stage_train(run, log):
    from algorithm2_benchmark.pipeline.train import train_final

    return train_final(run, log, spawner(run, log))


def stage_tune_gbdt(run, log):
    from algorithm2_benchmark.pipeline.train import tune_gbdt

    return tune_gbdt(run, log)


def stage_tune_tower(run, log):
    from algorithm2_benchmark.pipeline.train import tune_tower

    return tune_tower(run, log)


def stage_train_gbdt(run, log):
    from algorithm2_benchmark.pipeline.train import train_gbdt

    return train_gbdt(run, log)


def stage_train_tower(run, log):
    from algorithm2_benchmark.pipeline.train import train_tower

    return train_tower(run, log)


def stage_test_pass_gbdt(run, log):
    from algorithm2_benchmark.pipeline.final_eval import test_pass

    return test_pass(run, log, "gbdt")


def stage_test_pass_tower(run, log):
    from algorithm2_benchmark.pipeline.final_eval import test_pass

    return test_pass(run, log, "tower")


def stage_freeze(run, log):
    from algorithm2_benchmark.pipeline.select import freeze

    return freeze(run, log)


def stage_evaluate_test(run, log):
    from algorithm2_benchmark.pipeline.final_eval import evaluate_test

    return evaluate_test(run, log, spawner(run, log))


def stage_benchmark(run, log):
    from algorithm2_benchmark.pipeline.benchmark import benchmark

    return benchmark(run, log)


def stage_verify_integration(run, log):
    from algorithm2_benchmark.pipeline.integration import verify_integration

    return verify_integration(run, log)


def stage_report(run, log):
    from algorithm2_benchmark.pipeline.report import report

    return report(run, log)


STAGES = {"audit": stage_audit, "prepare": stage_prepare, "split": stage_split, "tune": stage_tune,
          "train": stage_train, "freeze": stage_freeze, "evaluate-test": stage_evaluate_test,
          "benchmark": stage_benchmark, "verify-integration": stage_verify_integration, "report": stage_report,
          # isolated sub-stages (spawned by tune / train / evaluate-test)
          "tune-gbdt": stage_tune_gbdt, "tune-tower": stage_tune_tower, "train-gbdt": stage_train_gbdt,
          "train-tower": stage_train_tower, "test-pass-gbdt": stage_test_pass_gbdt,
          "test-pass-tower": stage_test_pass_tower}
SUB_STAGES = ["tune-gbdt", "tune-tower", "train-gbdt", "train-tower", "test-pass-gbdt", "test-pass-tower"]


def _metadata(run, stage: str, started: str, status: str, error: str | None, seconds: float) -> None:
    from algorithm2_benchmark.pipeline.common import read_json

    path = run.path("run_metadata.json")
    meta = read_json(path) if path.exists() else {"run_id": run.run_id, "stages": []}
    meta.setdefault("start_git", git_state())
    meta["final_git"] = git_state()
    meta["environment"] = environment()
    meta["stages"].append({"stage": stage, "started_at": started, "finished_at": now(), "status": status,
                           "error": error, "seconds": round(seconds, 1), "peak_rss_mb": peak_rss_mb()})
    write_json(path, meta)


def run_stage(run, stage: str, log) -> None:
    started, t0 = now(), time.time()
    try:
        STAGES[stage](run, log)
    except BaseException as exc:
        _metadata(run, stage, started, "failed", f"{type(exc).__name__}: {exc}"[:500], time.time() - t0)
        raise
    _metadata(run, stage, started, "succeeded", None, time.time() - t0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m algorithm2_benchmark", description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=[*ORDER, "run-all", *SUB_STAGES])
    ap.add_argument("--run", required=True, help="run id (directory under algorithm2_benchmark/results/)")
    ap.add_argument("--level", choices=["fixture", "smoke", "main"], default=None)
    ap.add_argument("--root", default=None, help="runs root (default algorithm2_benchmark/results)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for LightGBM / PyTorch")
    args = ap.parse_args(argv)
    overrides = {}
    if args.threads:
        overrides = {"gbdt": {"num_threads": args.threads}, "two_tower": {"torch_threads": args.threads}}
    run = open_run(args.run, args.level, overrides, args.root)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    log = Logger(run.root / "work" / "run.log")
    if args.stage == "run-all":
        spawn = spawner(run, log, args.threads)          # every stage in a fresh interpreter
        for st in ORDER:
            if st not in ("audit", "benchmark", "verify-integration", "report") and stage_done(run.root, st):
                log(f"[run-all] {st}: already done, skipped")
                continue
            log(f"[run-all] {st} ...")
            spawn(st)
        return 0
    run_stage(run, args.stage, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
