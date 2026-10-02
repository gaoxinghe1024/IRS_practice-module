"""Algorithm 2 experiment end to end on the fixture level (generated data labelled synthetic/fixture; it
checks the logic, not recommendation quality): split and leakage control, real model fitting, identical
candidates for every model, offline / service agreement, frozen-run immutability and the written records.

The pipeline runs in a subprocess (`run-all` starts every stage in a fresh interpreter) because LightGBM
and PyTorch must not share a process; this test process only loads PyTorch and the NumPy tree reader.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from app.core.config import PROJECT_ROOT

pytestmark = pytest.mark.ml

RUN = "fx-test"


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    pytest.importorskip("torch")
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed (uv sync --extra ml)")
    root = tmp_path_factory.mktemp("alg2runs")
    r = subprocess.run([sys.executable, "-m", "algorithm2_benchmark", "run-all", "--run", RUN, "--level",
                        "fixture", "--root", str(root)], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    from algorithm2_benchmark.pipeline.config import open_run

    return open_run(RUN, None, None, root)


def _queries(run) -> pd.DataFrame:
    return pd.read_parquet(run.work / "query_manifest.parquet")


def test_partitions_are_user_disjoint_and_collections_stay_in_one_partition(fixture_run):
    users = pd.read_parquet(fixture_run.work / "users_split.parquet")
    assert users.groupby("offline_user_id")["partition"].nunique().max() == 1
    q = _queries(fixture_run)
    parts = {p: set(q[q.partition == p].offline_user_id) for p in ("train", "val", "test")}
    assert not parts["train"] & parts["val"] and not parts["train"] & parts["test"] and not parts["val"] & parts["test"]
    assert q.groupby("collection_id")["partition"].nunique().max() == 1
    assert q[q.partition != "train"].groupby("collection_id").size().max() == 1     # one hidden target in val / test


def test_hidden_target_is_never_part_of_the_profile_input(fixture_run):
    q = _queries(fixture_run)
    assert all(int(t) not in set(np.asarray(v).tolist()) for t, v in zip(q.target, q.visible, strict=True))
    assert (q.history_size == q.visible.map(len)).all()


def test_training_pairs_come_from_training_users_and_negatives_are_not_known_positives(fixture_run):
    pairs = pd.read_parquet(fixture_run.work / "sampled_pairs.parquet")
    q = _queries(fixture_run).set_index("query_id")
    users = pd.read_parquet(fixture_run.work / "users_split.parquet").set_index("offline_user_id")["partition"]
    train = pairs[pairs.partition == "train"]
    assert set(users.loc[q.loc[train.query_id.unique(), "offline_user_id"]]) == {"train"}
    pos = pd.read_parquet(fixture_run.work / "positives.parquet")
    known = pos.groupby("offline_user_id")["item"].apply(set).to_dict()
    for r in pairs[pairs.label == 0].itertuples(index=False):
        assert int(r.item) not in known[int(q.loc[r.query_id, "offline_user_id"])]
    # every learned model trains on the same pairs (one shared file, hashed in the split manifest)
    assert json.loads(fixture_run.path("split_manifest.json").read_text())["pairs"]["sha256"]


def test_every_model_ranks_the_same_candidates(fixture_run):
    pred = pd.read_parquet(fixture_run.path("predictions.parquet"))
    per = pred.groupby(["query_id", "rule"]).agg(hashes=("cand_hash", "nunique"), n=("n_cand", "nunique"),
                                                 models=("model", "nunique"))
    assert (per.hashes == 1).all() and (per.n == 1).all()
    assert set(pred.family) >= {"A0", "A1", "B0", "B1", "C0"} and (per.models == per.models.max()).all()
    assert int(pred.violations.sum()) == 0 and int(pred.dup_in_top.sum()) == 0 and not pred.relaxed.any()


def test_learned_models_were_really_fitted(fixture_run):
    mm = json.loads(fixture_run.path("models_manifest.json").read_text())
    curves = json.loads(fixture_run.path("training_curves.json").read_text())
    for m in mm["models"]:
        assert (fixture_run.root / m["file"]).stat().st_size > 0
        if m["family"] in ("B0", "B1"):
            assert m["best_iteration"] >= 1 and "Tree=0" in (fixture_run.root / m["file"]).read_text()
        else:
            hist = curves[f"C0_seed{m['seed']}"]
            assert hist["train_loss"][-1] < hist["train_loss"][0]                 # the loss went down


def test_lightgbm_learns_a_learnable_signal_in_a_separate_process():
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed")
    code = textwrap.dedent("""
        import numpy as np
        from algorithm2_benchmark.pipeline.train import train_lgbm
        from algorithm2_benchmark.pipeline.metrics import query_metrics
        rng = np.random.default_rng(0)
        X = rng.random((4000, 2)); y = (X[:, 0] + 0.1 * rng.standard_normal(4000) > 0.5).astype(int)
        g = {"n_estimators": 60, "early_stopping_rounds": 10, "num_threads": 1}
        p = {"learning_rate": 0.1, "num_leaves": 15, "max_depth": 5, "min_child_samples": 20, "reg_lambda": 1.0}
        b, it, curve, _ = train_lgbm(X[:3000], y[:3000], X[3000:], y[3000:], p, 42, g)
        s = b.predict(X[3000:], raw_score=True)
        pos, neg = s[y[3000:] == 1], s[y[3000:] == 0]
        auc = (pos[:, None] > neg[None, :]).mean()
        assert it >= 1 and b.num_trees() >= 1 and curve["val_logloss"][-1] < curve["val_logloss"][0], curve
        assert auc > 0.9, auc
        print("ok", it, round(float(auc), 3))
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stdout + r.stderr


def test_two_tower_takes_real_gradient_steps():
    torch = pytest.importorskip("torch")
    from app.recommendation.algorithm2.two_tower import train_two_tower

    rng = np.random.default_rng(0)
    coll = rng.random((200, 6)).astype(np.float32)
    items = rng.random((50, 5)).astype(np.float32)
    q = rng.integers(0, 200, 3000)
    i = rng.integers(0, 50, 3000)
    y = ((coll[q, 0] > 0.5) == (items[i, 0] > 0.5)).astype(np.float32)        # learnable agreement signal
    cfg = {"out_dim": 8, "dropout": 0.0, "learning_rate": 3e-3, "weight_decay": 0.0, "batch_size": 256,
           "max_epochs": 15, "early_stopping_patience": 5, "hidden": [16, 8]}
    torch.manual_seed(0)
    model, hist = train_two_tower(coll, items, (q[:2500], i[:2500], y[:2500]), (q[2500:], i[2500:], y[2500:]), cfg, 42,
                                  log=lambda *_: None)
    assert hist["train_loss"][-1] < hist["train_loss"][0] - 0.05
    assert min(hist["val_loss"]) < hist["val_loss"][0]


def test_service_top8_equals_the_offline_prediction(fixture_run):
    from algorithm2_benchmark.pipeline.data import make_fixture
    from algorithm2_benchmark.pipeline.evaluate import RULE_IDS, RuleMasks
    from app.recommendation.algorithm2.bundle import load_scorers
    from app.recommendation.algorithm2.service import Alg2Service

    cfg = fixture_run.config()
    prep = make_fixture()
    fz = json.loads(fixture_run.path("freeze_manifest.json").read_text())
    seed = fz["deploy_seed"]
    labels = [f"{f}_seed{seed}" for f in ("B0", "B1", "C0")]
    loaded, _ = load_scorers(fixture_run.root, prep.cat, labels, 1)            # NumPy tree reader, as served
    scorers = {"A0": loaded["A0"], "A1": loaded["A1"], **{lab.split("_seed")[0]: loaded[lab] for lab in labels}}
    svc = Alg2Service(prep.cat, prep.engine, scorers, "A0", "ready", "test")
    rules = RuleMasks(prep.cat, prep.engine, cfg)
    pred = pd.read_parquet(fixture_run.path("predictions.parquet"))
    offline = {(r.query_id, r.rule, r.model): r.top_keys for r in pred.itertuples(index=False)}
    q = _queries(fixture_run)
    compared = 0
    for row in q[q.partition == "test"].itertuples(index=False):
        for rule in RULE_IDS:
            cs = rules.constraint_set(rule, row.r1_keyword)
            if cs is None:
                continue
            for fam in ("A0", "A1", "B0", "B1", "C0"):
                res = svc.recommend(np.asarray(row.visible, dtype=np.int64), cs, fam, 8)
                label = fam if fam in ("A0", "A1") else f"{fam}_seed{seed}"
                expect = [k for k in (offline[(row.query_id, rule, label)] or "").split("|") if k]
                assert prep.cat.snap.keys[res.items].tolist() == expect, (row.query_id, rule, fam)
                compared += 1
    assert compared > 50


def test_frozen_run_cannot_be_retuned_and_reevaluation_reproduces(fixture_run):
    def cli(stage):
        return subprocess.run([sys.executable, "-m", "algorithm2_benchmark", stage, "--run", RUN, "--root",
                               str(fixture_run.root.parent)], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              timeout=900)

    for stage in ("tune", "train"):
        r = cli(stage)
        assert r.returncode != 0 and "frozen" in (r.stdout + r.stderr)
    before = json.loads(fixture_run.path("metrics.json").read_text())
    assert cli("evaluate-test").returncode == 0
    after = json.loads(fixture_run.path("metrics.json").read_text())
    for key in ("primary", "families", "end_to_end", "rule_eligibility"):
        assert before[key] == after[key]
    assert before["decision"]["default_scorer"] == after["decision"]["default_scorer"]


def test_records_and_reports_are_written(fixture_run):
    for name in ("data_manifest.json", "split_manifest.json", "selection_protocol.json", "feature_schema.json",
                 "freeze_manifest.json", "run_metadata.json", "trial_results.csv", "summary.csv", "metrics.json",
                 "predictions.parquet", "subgroup_metrics.csv", "paired_comparisons.csv", "latency.csv",
                 "training_curves.json", "feature_importance.csv", "models_manifest.json", "case_analysis.md",
                 "final_report.md", "summary_en.md", "README_reproduce.md", "integration_report.json"):
        assert fixture_run.path(name).exists(), name
    assert json.loads(fixture_run.path("integration_report.json").read_text())["status"] == "not_applicable"
    assert "synthetic/fixture" in fixture_run.path("summary.csv").read_text()
