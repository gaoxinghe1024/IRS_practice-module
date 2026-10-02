"""`audit` stage: machine-checkable repository and data facts behind docs/algorithm2/repo_audit.md."""

from __future__ import annotations

import sqlite3
import subprocess

from app.core.config import PROJECT_ROOT, load_settings
from evaluation.algorithm2.common import git_state, mark_stage, now, write_json

CODE_LOCATIONS = {
    "collection_page": ["app/web/templates/collection.html", "app/web/static/js/pages/collection.js"],
    "api": ["app/api/v1/collections.py::recommendations"],
    "service": ["app/services/recommend.py::collection"],
    "rules": ["app/recommendation/constraints.py::ConstraintEngine"],
    "algorithm2": ["app/recommendation/algorithm2/"],
    "legacy_scorers": ["app/recommendation/collection/manual.py", "app/recommendation/profiles.py"],
    "data_import": ["pipelines/real_ingest.py", "pipelines/activate.py"],
}


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def audit(run, log) -> dict:
    cfg = run.config()
    facts = {"created_at": now(), "git": git_state(), "remotes": _git("remote", "-v").splitlines(),
             "origin_main": _git("rev-parse", "origin/main"), "upstream": _git("rev-parse", "--abbrev-ref", "@{u}"),
             "code_locations": CODE_LOCATIONS,
             "code_locations_exist": {k: all((PROJECT_ROOT / p.split("::")[0]).exists() for p in v)
                                      for k, v in CODE_LOCATIONS.items()}}
    try:
        settings = load_settings("real", environ={})
        facts["configured_collection"] = settings.recommendation.get("collection", {})
        db = settings.db_path
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            q = {"anime": "SELECT count(*) FROM anime", "anime_in_catalog": "SELECT count(*) FROM anime WHERE in_catalog=1",
                 "users": "SELECT count(*) FROM users", "collections": "SELECT count(*) FROM collections",
                 "collection_items": "SELECT count(*) FROM collection_items", "app_ratings": "SELECT count(*) FROM ratings"}
            facts["business_db"] = {"path": str(db.relative_to(PROJECT_ROOT)),
                                    **{k: con.execute(v).fetchone()[0] for k, v in q.items()}}
            con.close()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        facts["business_db_error"] = f"{type(exc).__name__}: {exc}"
    if cfg["level"] != "fixture":
        from evaluation.algorithm2.data import active_real_version

        facts["active_real_version"] = active_real_version()
    write_json(run.path("audit_facts.json"), facts)
    mark_stage(run.root, "audit", {})
    log(f"[audit] branch={facts['git']['branch']} head={facts['git']['commit'][:9]} origin/main={facts['origin_main'][:9]}")
    return facts
