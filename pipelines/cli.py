"""AniLogic command line: data pipelines, training, indexing, activation, evaluation, admin tasks.

    python -m pipelines.cli <command> [--mode synthetic|real] [options]

Exit codes: 0 success, 2 invalid / missing input data, 3 precondition not met
(e.g. nothing to activate), 1 unexpected failure. Every command is recorded
under var/<mode>/pipeline_runs/.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.core.config import Settings, load_settings
from pipelines.common import pipeline_run


def _settings(args) -> Settings:
    return load_settings(getattr(args, "mode", None))


def cmd_generate_synthetic(args):
    from pipelines.synthetic import generate

    s = load_settings("synthetic")
    if args.seed is not None:
        s.synthetic["seed"] = args.seed
    with pipeline_run("generate-synthetic", s) as rec:
        rec.outputs = generate(s)["counts"]


def cmd_validate_real(args):
    from pipelines.real_ingest import validate_real

    s = load_settings("real")
    with pipeline_run("validate-real", s) as rec:
        report = validate_real(s, count_rows=not args.fast)
        rec.outputs = {"ok": report["ok"], "errors": report["errors"], "warnings": report["warnings"]}
    print(json.dumps({k: report[k] for k in ("ok", "errors", "warnings")}, indent=2))
    return 0 if report["ok"] else 2


def cmd_validate_data(args):
    from pipelines.validate_data import validate_all

    s = _settings(args)
    out = validate_all(s)
    print(json.dumps(out, indent=2))
    return 0 if all(v.get("ok", True) for v in out.values()) else 2


def cmd_preprocess(args):
    s = _settings(args)
    with pipeline_run("preprocess", s) as rec:
        if s.mode == "real":
            from pipelines.real_ingest import preprocess_real

            m = preprocess_real(s)
        else:
            from pipelines.synthetic import preprocess_synthetic

            m = preprocess_synthetic(s)
        rec.outputs = {"dataset_version": m["dataset_version"], "counts": m["counts"]}


def cmd_split(args):
    from pipelines.split import make_splits

    s = _settings(args)
    with pipeline_run("split", s) as rec:
        m = make_splits(s, args.version)
        rec.outputs = {k: {"counts": v["counts"], "fingerprint": v["fingerprint"]} for k, v in m.items()}


def cmd_train(args):
    from pipelines.fit import ALL_COMPONENTS, train

    s = _settings(args)
    comps = tuple(args.components.split(",")) if args.components else ALL_COMPONENTS
    with pipeline_run("train", s) as rec:
        m = train(s, args.version, comps)
        rec.outputs = {k: v.get("status") for k, v in m.get("components", {}).items()}


def cmd_build_index(args):
    from pipelines.build_index import build_index

    s = _settings(args)
    with pipeline_run("build-index", s) as rec:
        idx = build_index(s, args.version)
        rec.outputs = {"documents": idx["documents"], "dense": idx["dense"].get("provider")}


def cmd_activate(args):
    from pipelines.activate import ActivationError, activate

    s = _settings(args)
    try:
        with pipeline_run("activate", s) as rec:
            rec.outputs = activate(s, args.version)
    except ActivationError as exc:
        print(f"activation refused: {exc}", file=sys.stderr)
        return 3


def cmd_build_all(args):
    """generate (synthetic) -> preprocess -> split -> train -> build-index -> activate [-> seed-demo]."""
    s = _settings(args)
    if s.mode == "synthetic":
        cmd_generate_synthetic(argparse.Namespace(seed=None))
    elif cmd_validate_real(argparse.Namespace(fast=True)) != 0:
        return 2
    for fn in (cmd_preprocess, cmd_split, cmd_train, cmd_build_index, cmd_activate):
        rc = fn(argparse.Namespace(mode=s.mode, version=None, components=None))
        if rc:
            return rc
    if s.mode == "synthetic" and not args.no_demo:
        cmd_seed_demo(argparse.Namespace(mode="synthetic"))
    return 0


def cmd_seed_demo(args):
    from pipelines.demo_seed import seed_demo

    s = load_settings("synthetic")
    with pipeline_run("seed-demo", s) as rec:
        rec.outputs = seed_demo(s)


def cmd_make_admin(args):
    from sqlalchemy import select

    from app.db.models import User
    from app.db.session import session_scope
    from app.services.auth import normalize_username

    s = _settings(args)
    with session_scope(s.db_url) as db:
        u = db.scalar(select(User).where(User.normalized_username == normalize_username(args.username)))
        if u is None:
            print("no such user", file=sys.stderr)
            return 3
        u.is_admin = True
    print(f"{args.username} is now an administrator in the {s.mode} database")


def cmd_enrich_jikan(args):
    from pipelines.jikan import enrich

    s = load_settings("real")
    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    with pipeline_run("enrich-jikan", s) as rec:
        rec.outputs = enrich(s, args.top_popular, ids, not args.no_characters, args.force)


def cmd_migrate_legacy(args):
    from pipelines.legacy_migration import migrate

    s = _settings(args)
    with pipeline_run("migrate-legacy", s) as rec:
        r = migrate(s, args.legacy_db, apply=args.apply, include_seed_pattern=args.include_seed_pattern)
        rec.outputs = {"users": len(r["users"]), "skipped_users": len(r["skipped_users"]),
                       "unmapped": len(r["unmapped"]), "title_corrections": len(r["title_corrections"]),
                       "apply": args.apply}


def cmd_import_package(args):
    from pipelines.package_import import import_package

    s = load_settings("real")
    package = args.package or s.real.get("team_package_dir")
    if not package:
        raise FileNotFoundError("no package path: pass --package or set real.team_package_dir")
    with pipeline_run("import-package", s) as rec:
        r = import_package(s, s._abs(str(package)), force=args.force)
        rec.outputs = {"catalog_version": r["catalog_version"], **r["written"]}


def cmd_import_similarity_labels(args):
    from pipelines.package_import import import_similarity_labels

    s = load_settings("real")
    with pipeline_run("import-similarity-labels", s) as rec:
        rec.outputs = import_similarity_labels(s, args.file)


def cmd_evaluate(args):
    from evaluation.runner import run_experiments

    s = _settings(args)
    with pipeline_run("evaluate", s) as rec:
        rec.outputs = run_experiments(s, args.experiment.split(","), sample_users=args.sample_users,
                                      seeds=args.seeds, tag=args.tag)


def cmd_export_evaluation(args):
    from evaluation.reports import export_summary

    s = _settings(args)
    print(export_summary(s))


def cmd_status(args):
    from app.services.serving import load_state

    s = _settings(args)
    st = load_state(s)
    print(json.dumps({"mode": s.mode, "dataset_version": st.dataset_version,
                      "capabilities": {k: v.public() for k, v in st.capabilities.items()}}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anilogic", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, fn, mode=True, **kw):
        p = sub.add_parser(name, **kw)
        if mode:
            p.add_argument("--mode", choices=["synthetic", "real"], default=None,
                           help="dataset mode (default: ANILOGIC_DATASET_MODE or configs/base.yaml)")
        p.set_defaults(fn=fn)
        return p

    p = add("generate-synthetic", cmd_generate_synthetic, mode=False, help="generate the seeded fictional dataset")
    p.add_argument("--seed", type=int, default=None)
    p = add("validate-real", cmd_validate_real, mode=False, help="check Kaggle files, headers, hashes, row counts")
    p.add_argument("--fast", action="store_true", help="skip full row counts")
    add("validate-data", cmd_validate_data, help="validate optional enriched / evaluation JSONL contracts")
    add("preprocess", cmd_preprocess, help="clean + build the processed dataset version")
    for name, fn in (("split", cmd_split), ("train", cmd_train), ("build-index", cmd_build_index),
                     ("activate", cmd_activate)):
        p = add(name, fn)
        p.add_argument("--version", default=None, help="dataset version (default: latest)")
        if name == "train":
            p.add_argument("--components", default=None, help="comma list: icf,mf,ncf,embeddings,logistic,gbdt,two_tower")
    p = add("build-all", cmd_build_all, help="run the whole pipeline for a mode")
    p.add_argument("--no-demo", action="store_true", help="synthetic: skip demo accounts")
    add("seed-demo", cmd_seed_demo, mode=False, help="create documented demo accounts (synthetic only)")
    p = add("make-admin", cmd_make_admin, help="grant admin to an existing account")
    p.add_argument("username")
    p = add("enrich-jikan", cmd_enrich_jikan, mode=False, help="offline Jikan enrichment (real mode)")
    p.add_argument("--top-popular", type=int, default=300)
    p.add_argument("--ids", default=None, help="comma separated MAL ids (overrides --top-popular)")
    p.add_argument("--no-characters", action="store_true")
    p.add_argument("--force", action="store_true")
    p = add("migrate-legacy", cmd_migrate_legacy, help="migrate accounts from backend/anime.db (dry run default)")
    p.add_argument("--legacy-db", default="backend/anime.db")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--include-seed-pattern", action="store_true")
    p = add("import-package", cmd_import_package, mode=False,
            help="convert the team's curated data package into supplementary inputs (real mode)")
    p.add_argument("--package", default=None, help="package root (contains DATASET_INDEX.json)")
    p.add_argument("--force", action="store_true", help="replace an existing data/evaluation/rag_queries.jsonl")
    p = add("import-similarity-labels", cmd_import_similarity_labels, mode=False,
            help="MAL-recommendation ground truth of the algorithm-1 benchmark -> weak similarity labels")
    p.add_argument("--file", default="algorithm1_benchmark/data/mal_ground_truth.jsonl")
    p = add("evaluate", cmd_evaluate, help="run evaluation experiments")
    p.add_argument("--experiment", default="all",
                   help="comma list: similar,collection,homepage,ga_ablation,explicit,chat,planner,system or all")
    p.add_argument("--sample-users", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--tag", default=None)
    add("export-evaluation", cmd_export_evaluation, help="summarise the latest evaluation runs")
    add("status", cmd_status, help="print capability status for the active version")
    args = ap.parse_args(argv)
    try:
        rc = args.fn(args)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - reported and non-zero
        from pipelines.real_ingest import DataValidationError

        if isinstance(exc, DataValidationError):
            print(f"invalid input data: {exc}", file=sys.stderr)
            return 2
        raise
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
