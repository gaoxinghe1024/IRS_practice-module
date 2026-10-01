"""Evaluation run bookkeeping: every experiment writes a self-describing run directory.

evaluation/runs/<run_id>/
    config.yaml       exact settings snapshot + experiment parameters
    manifest.json     dataset mode/version, split fingerprints, artifact manifest hash, seeds, hardware, git commit
    metrics.json      aggregate metrics (with 95% bootstrap CIs and sample sizes)
    per_case.jsonl    one row per evaluated user / query / case
    predictions.csv   ranked outputs per case (local only; git-ignored)
    failures.jsonl    cases that failed or were excluded, with the reason
    comparison.csv    method-by-method table
    report.md         human-readable report (data, protocol, results, limitations)
Experiments that cannot run are recorded with status not_run / blocked_* and no numbers.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.config import PROJECT_ROOT, Settings
from pipelines.common import dependency_versions, peak_rss_mb, read_json, sha256_file, utc_now_iso, write_json


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}


@dataclass
class Run:
    settings: Settings
    experiment: str
    params: dict
    run_id: str = ""
    root: Path = None
    started: float = field(default_factory=time.time)
    status: str = "completed"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        tag = self.params.get("tag")
        self.run_id = (time.strftime("%Y%m%d-%H%M%S") + f"-{self.settings.mode}-{self.experiment}"
                       + (f"-{tag}" if tag else "") + "-" + uuid.uuid4().hex[:4])
        self.root = self.settings.evaluation_runs_dir / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "config.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump({"experiment": self.experiment, "params": self.params,
                            "settings": json.loads(self.settings.model_dump_json())}, fh, sort_keys=False)
        self._cases = (self.root / "per_case.jsonl").open("w", encoding="utf-8")
        self._fails = (self.root / "failures.jsonl").open("w", encoding="utf-8")

    def case(self, row: dict) -> None:
        self._cases.write(json.dumps(row, default=_jsonable) + "\n")

    def failure(self, row: dict) -> None:
        self._fails.write(json.dumps(row, default=_jsonable) + "\n")

    def predictions(self, rows: list[dict]) -> None:
        if not rows:
            return
        with (self.root / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def comparison(self, rows: list[dict]) -> None:
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (self.root / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def finish(self, metrics: dict, report_md: str, data_manifest: dict) -> dict:
        self._cases.close()
        self._fails.close()
        from pipelines.common import latest_dataset_version

        try:
            version = latest_dataset_version(self.settings)
        except FileNotFoundError:
            version = None
        art_manifest = self.settings.artifacts_dir / version / "manifest.json" if version else None
        manifest = {"run_id": self.run_id, "experiment": self.experiment, "status": self.status,
                    "dataset_mode": self.settings.mode, "dataset_version": version,
                    "artifact_manifest_sha256": sha256_file(art_manifest) if art_manifest and art_manifest.exists()
                    else None,
                    "split_fingerprints": _split_fps(self.settings, version),
                    "params": self.params, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                    "finished_at": utc_now_iso(), "seconds": round(time.time() - self.started, 1),
                    "peak_rss_mb": peak_rss_mb(), "git_commit": git_commit(), "hardware": hardware(),
                    "dependencies": dependency_versions(), "data": data_manifest, "notes": self.notes}
        if self.params.get("tag") == "smoke":
            manifest["smoke"] = True
            head, _, rest = report_md.partition("\n")
            report_md = (f"{head}\n\n> **Smoke run** (small samples, one seed): checks that the evaluation harness "
                         f"runs end to end. Not a quality result; `make evaluate` produces the reported numbers.\n{rest}")
        write_json(self.root / "manifest.json", manifest)
        write_json(self.root / "metrics.json", {"status": self.status, **metrics})
        (self.root / "report.md").write_text(report_md, encoding="utf-8")
        return {"run_id": self.run_id, "status": self.status, "dir": str(self.root)}


def _split_fps(settings: Settings, version: str | None) -> dict:
    out = {}
    if not version:
        return out
    for proto in ("ranking", "explicit"):
        p = settings.processed_dir / version / "splits" / proto / "manifest.json"
        if p.exists():
            out[proto] = read_json(p).get("fingerprint")
    return out


def _jsonable(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def ci_text(ci: dict) -> str:
    if not ci or ci.get("mean") is None:
        return "–"
    return f"{ci['mean']:.4f} [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] (n={ci['n']})"
