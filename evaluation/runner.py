"""Dispatch evaluation experiments (see docs/evaluation-protocol.md)."""

from __future__ import annotations

import importlib
import traceback

from app.core.config import Settings

ORDER = ["similar", "collection", "homepage", "explicit", "ga_ablation", "chat", "planner", "system", "multimodal"]


def run_experiments(settings: Settings, names: list[str], sample_users: int | None = None, seeds: int | None = None,
                    tag: str | None = None, log=print) -> dict:
    names = ORDER if names in (["all"], ["*"]) else names
    unknown = [n for n in names if n not in ORDER]
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}; choose from {ORDER}")
    out = {}
    for name in names:
        log(f"[evaluate] {name} ({settings.mode}) ...")
        mod = importlib.import_module(f"evaluation.experiments.{name}")
        kwargs = {"tag": tag}
        if sample_users is not None:
            kwargs["sample_users_n"] = sample_users
        if seeds is not None:
            kwargs["seeds"] = seeds
        try:
            res = mod.run(settings, **kwargs)
        except Exception as exc:  # recorded, the other experiments still run
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback_tail": traceback.format_exc().splitlines()[-5:]}
        out[name] = res
        log(f"[evaluate] {name}: {res.get('status')} {res.get('dir', res.get('error', ''))}")
    return out
