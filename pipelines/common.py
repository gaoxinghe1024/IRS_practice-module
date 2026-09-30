"""Shared pipeline utilities: hashing, manifests, versions and run records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, Settings

PIPELINE_VERSION = "2.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str, sort_keys=False)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc


def write_jsonl(path: Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def dependency_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = ["numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "fastapi", "sqlalchemy",
             "torch", "sentence-transformers"]
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not-installed"
    return out


def dataset_version(mode: str, source_hashes: dict[str, str], config_fp: str) -> str:
    digest = sha256_json({"mode": mode, "sources": source_hashes, "pipeline": PIPELINE_VERSION, "config": config_fp})
    return f"{mode}-{digest[:10]}"


def processed_root(settings: Settings, version: str) -> Path:
    return settings.processed_dir / version


def active_pointer_path(settings: Settings) -> Path:
    return settings.artifacts_dir / "active.json"


def latest_pointer_path(settings: Settings) -> Path:
    return settings.processed_dir / "latest.json"


def latest_dataset_version(settings: Settings) -> str:
    p = latest_pointer_path(settings)
    if not p.exists():
        raise FileNotFoundError(
            f"No processed dataset for mode '{settings.mode}'. Run the preprocess step first "
            f"(make seed for synthetic, make preprocess-real for real).")
    return read_json(p)["dataset_version"]


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class RunRecord:
    command: str
    settings: Settings
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    input_hashes: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    error_summary: str | None = None

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "command": self.command, "dataset_mode": self.settings.mode,
                "config_hash": self.settings.fingerprint(), "input_hashes": self.input_hashes,
                "outputs": self.outputs, "status": self.status, "started_at": self.started_at,
                "finished_at": self.finished_at, "error_summary": self.error_summary,
                "peak_rss_mb": peak_rss_mb()}


@contextmanager
def pipeline_run(command: str, settings: Settings) -> Iterator[RunRecord]:
    """Record a pipeline command under var/<mode>/pipeline_runs/<run_id>.json."""
    rec = RunRecord(command=command, settings=settings)
    out = settings.var_dir / "pipeline_runs" / f"{rec.run_id}-{command}.json"
    try:
        yield rec
        rec.status = "succeeded"
    except BaseException as exc:
        rec.status = "failed"
        rec.error_summary = f"{type(exc).__name__}: {exc}"[:2000]
        rec.outputs.setdefault("traceback_tail", traceback.format_exc().splitlines()[-6:])
        raise
    finally:
        rec.finished_at = utc_now_iso()
        write_json(out, rec.to_dict())
