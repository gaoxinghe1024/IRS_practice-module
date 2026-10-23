"""Copy validation-tuned ICF parameters from the latest homepage evaluation run into configs/<mode>.yaml.

    python -m pipelines.apply_tuning --mode real

Only validation results are used (the homepage experiment tunes on validation positives of a disjoint
user group). The config line records the run id, so the served value is traceable to its evidence.
"""

from __future__ import annotations

import argparse
import json
import re

from app.core.config import CONFIG_DIR, load_settings

# metrics key -> (config section under recommendation, config key, metrics field)
TUNED = [("icf_served_params", "icf", "prediction_damping", "damping"),
         ("icf_served_params", "icf", "user_mean_prior_weight", "user_mean_prior_weight"),
         ("home_served_params", "home", "content_popularity_weight", "content_popularity_weight")]


def latest_homepage(settings) -> tuple[str, dict]:
    runs = sorted(d for d in settings.evaluation_runs_dir.iterdir()
                  if d.is_dir() and f"-{settings.mode}-homepage-" in d.name and "-smoke-" not in d.name
                  and (d / "metrics.json").exists())
    if not runs:
        raise SystemExit(f"no finished homepage run for mode {settings.mode}")
    return runs[-1].name, json.loads((runs[-1] / "metrics.json").read_text())


def _set(text: str, section: str, key: str, value: float, run_id: str) -> str:
    line = f"    {key}: {value}  # validation-tuned, evaluation run {run_id}\n"
    text = re.sub(rf"^    {key}:.*\n", "", text, flags=re.MULTILINE)
    if re.search(rf"^  {section}:\n", text, flags=re.MULTILINE):
        return re.sub(rf"^  {section}:\n", lambda m: m.group(0) + line, text, count=1, flags=re.MULTILINE)
    if re.search(r"^recommendation:\n", text, flags=re.MULTILINE):
        return re.sub(r"^recommendation:\n", lambda m: m.group(0) + f"  {section}:\n" + line, text, count=1,
                      flags=re.MULTILINE)
    return text + f"\nrecommendation:\n  {section}:\n" + line


def apply(mode: str) -> dict:
    settings = load_settings(mode)
    run_id, metrics = latest_homepage(settings)
    path = CONFIG_DIR / f"{mode}.yaml"
    text = path.read_text(encoding="utf-8")
    applied = {}
    for mkey, section, key, field in TUNED:
        value = (metrics.get(mkey) or {}).get(field)
        if value is None:
            continue
        text = _set(text, section, key, float(value), run_id)
        applied[f"{section}.{key}"] = float(value)
    if not applied:
        raise SystemExit(f"{run_id} has no tuned parameters")
    path.write_text(text, encoding="utf-8")
    check = load_settings(mode).recommendation
    for name, value in applied.items():
        section, key = name.split(".")
        assert float(check[section][key]) == value, name
    print(f"[apply-tuning] {mode}: {applied} from {run_id} -> {path.name}")
    return applied


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["synthetic", "real"], required=True)
    apply(ap.parse_args().mode)
