"""Run configuration and run-directory layout for the algorithm-2 experiments.

Every value that can change a result is fixed here (or in the run's config.json) before data is read:
split / negative seeds, label threshold, rule scenarios, search spaces and the selection thresholds.
A run directory keeps small, committed records at its top level and large regenerable files under
`work/` (git-ignored).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.recommendation.algorithm2.catalog import DEFAULT_SUPPLEMENT

RUNS_ROOT = PROJECT_ROOT / "algorithm2_benchmark" / "results"
LEVELS = ("fixture", "smoke", "main")

DEFAULTS: dict = {
    "level": "main",
    "data": {
        "mode": "real",                    # fixture level ignores this and uses generated fixture data
        "dataset_version": None,           # None -> the active real version (artifacts/real/active.json)
        "positive_threshold": 8,           # personal rating >= 8 forms the positive preference set (proxy label)
        "supplement": DEFAULT_SUPPLEMENT,  # theme / demographic projection keyed by MAL id
    },
    "split": {
        "split_seed": 42,
        "fractions": [0.70, 0.15, 0.15],
        "main_min_positive": 6,            # main group: positive set >= 6 (visible history >= 5)
        "sparse_range": [2, 5],            # sparse group: 2-5 (visible history 1-4), reported separately
        "main_users": 20000,
        "sparse_users": 4000,
        "smoke_main_users": 1000,          # nested subset of the main split
        "smoke_sparse_users": 200,
        "train_targets_per_collection": 3,
        "size_buckets": [6, 10, 20, 50, 100],
    },
    "rules": {"R2": {"min_score": 7.0, "max_episodes": 26}},
    "negatives": {"negative_seed": 42, "per_query": 4},
    "k": 8,
    "a1_weights": [0.05, 0.10, 0.15, 0.20, 0.30],
    "gbdt": {
        "max_trials": 10,
        "grid": {"num_leaves": [15, 31], "max_depth": [5, 8], "learning_rate": [0.03, 0.05],
                 "min_child_samples": [50, 100], "reg_lambda": [1.0, 5.0]},
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "num_threads": 8,
        "trial_seed": 42,
    },
    "two_tower": {
        "max_trials": 10,
        "grid": {"out_dim": [32, 64], "dropout": [0.0, 0.2], "learning_rate": [1e-3, 3e-4],
                 "weight_decay": [1e-5, 1e-4]},
        "hidden": [128, 64],
        "batch_size": 512,
        "max_epochs": 30,
        "early_stopping_patience": 3,
        "device": "cpu",
        "studio_min_count": 5,
        "torch_threads": 8,
        "trial_seed": 42,
    },
    "seeds": [42, 2026, 7],
    "deploy_seed": 42,
    "selection": {
        "tie_margin": 0.005,               # 0.5 percentage points
        "auc_min": 0.65,
        "hr_gain_min": 0.02,               # 2 percentage points vs A0
        "bootstrap": 2000,
        "bootstrap_seed": 42,
    },
    "latency": {"warmup": 20, "requests": 200, "threads": 1},
}

LEVEL_OVERRIDES: dict = {
    "fixture": {"split": {"main_users": 60, "sparse_users": 30, "smoke_main_users": 60, "smoke_sparse_users": 30},
                "gbdt": {"max_trials": 3, "n_estimators": 60, "early_stopping_rounds": 10, "num_threads": 2},
                "two_tower": {"max_trials": 2, "max_epochs": 4, "batch_size": 64, "torch_threads": 2},
                "latency": {"warmup": 2, "requests": 10}, "selection": {"bootstrap": 200}},
    "smoke": {},
    "main": {},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build_config(level: str, overrides: dict | None = None) -> dict:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS}")
    cfg = _merge(DEFAULTS, LEVEL_OVERRIDES[level])
    cfg["level"] = level
    return _merge(cfg, overrides or {})


@dataclass
class Run:
    run_id: str
    root: Path

    @property
    def work(self) -> Path:
        p = self.root / "work"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models(self) -> Path:
        p = self.root / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, name: str) -> Path:
        return self.root / name

    def config(self) -> dict:
        return json.loads((self.root / "config.json").read_text())

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def frozen(self) -> bool:
        return (self.root / "freeze_manifest.json").exists()


def open_run(run_id: str, level: str | None = None, overrides: dict | None = None, root: Path | None = None) -> Run:
    """Create (with config.json) or open a run. An existing run keeps its stored configuration."""
    base = Path(root) if root else RUNS_ROOT
    run = Run(run_id, base / run_id)
    cfg_path = run.root / "config.json"
    if cfg_path.exists():
        stored = json.loads(cfg_path.read_text())
        if level and stored.get("level") != level:
            raise SystemExit(f"run {run_id} was created with level {stored.get('level')!r}, not {level!r}")
        return run
    if not level:
        raise SystemExit(f"run {run_id} does not exist yet: pass --level fixture|smoke|main to create it")
    run.root.mkdir(parents=True, exist_ok=True)
    cfg = build_config(level, overrides)
    cfg["run_id"] = run_id
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return run
