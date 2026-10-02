"""Small helpers shared by the algorithm-2 stages: hashing, records, stage markers, logging, metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import PROJECT_ROOT


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_df(df: pd.DataFrame) -> str:
    """Content hash of a table (column names + values, row order included)."""
    h = hashlib.sha256(json.dumps(list(df.columns)).encode())
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            h.update(json.dumps([x.tolist() if isinstance(x, np.ndarray) else x for x in s.tolist()],
                                default=str).encode())
        else:
            h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def clean_float(x, nd: int = 6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.rstrip("\n")   # keep the leading status columns
        except Exception:  # noqa: BLE001 - git missing
            return ""
    status = run("status", "--porcelain")
    code_changes = [ln for ln in status.splitlines() if not ln[3:].startswith(("var/", "evaluation/algorithm2/runs/"))]
    diff = run("diff", "HEAD", "--", "app", "evaluation", "pipelines", "configs", "tests")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_code_files": [ln[3:] for ln in code_changes],
            "code_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def environment() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count()}
    from importlib.metadata import PackageNotFoundError, version

    # versions from package metadata: importing LightGBM and PyTorch together would load both OpenMP runtimes
    for mod, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("lightgbm", "lightgbm"),
                      ("torch", "torch"), ("sklearn", "scikit-learn"), ("pyarrow", "pyarrow")):
        try:
            out[mod] = version(dist)
        except PackageNotFoundError:
            out[mod] = None
    try:
        out["memory_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                                 timeout=5).stdout.strip() or 0) or None
    except Exception:  # noqa: BLE001 - not macOS
        out["memory_bytes"] = None
    return out


def peak_rss_mb() -> float | None:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)
    except Exception:  # noqa: BLE001
        return None


def mark_stage(run_root: Path, stage: str, info: dict) -> None:
    write_json(run_root / "stages" / f"{stage}.json", {"stage": stage, "finished_at": now(), **info})


def stage_done(run_root: Path, stage: str) -> dict | None:
    p = run_root / "stages" / f"{stage}.json"
    return read_json(p) if p.exists() else None
