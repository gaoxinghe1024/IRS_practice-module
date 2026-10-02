"""Algorithm 2 checks that need no trained models: metrics, tie-breaks, rules, profile isolation, strict
service behaviour and bundle compatibility. Training, leakage and offline / online agreement are checked
on a fixture run in tests/integration/test_algorithm2_pipeline.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.core.text import split_keywords
from app.recommendation.algorithm2.bundle import BundleMismatch, load_scorers
from app.recommendation.algorithm2.catalog import build_catalog
from app.recommendation.algorithm2.features import FEATURE_VERSION, compute_features, feature_schema
from app.recommendation.algorithm2.profiles import build_profile
from app.recommendation.algorithm2.scorers import FormulaScorer, rank_order, top_k
from app.recommendation.algorithm2.service import Alg2Service, build_service
from app.recommendation.constraints import ConstraintSet, build_engine
from evaluation.algorithm2.metrics import paired_bootstrap, query_metrics, user_macro
from tests.helpers import make_snapshot

ROWS = [
    {"title": "Alpha Mystery", "genre": ["Mystery"], "theme": ["Detective"], "studio": ["Bones"], "episodes": 12,
     "score": 8.5},
    {"title": "Beta Comedy", "genre": ["Comedy"], "episodes": 24, "score": 7.0},
    {"title": "Gamma", "genre": ["Mystery", "Horror"], "episodes": None, "score": 9.0},
    {"title": "Delta", "genre": ["Comedy"], "episodes": 12, "score": 6.99},
    {"title": "Epsilon", "genre": ["Mystery"], "episodes": 27, "score": 7.5},
    {"title": "Zeta Unscored", "genre": ["Mystery"], "episodes": 12, "score": None},
    {"title": "Eta Explicit", "genre": ["Mystery"], "episodes": 12, "score": 8.0, "is_explicit": True,
     "in_catalog": False},
    {"title": "Theta Mystery Comedy", "genre": ["Mystery", "Comedy"], "episodes": 13, "score": 7.2},
    {"title": "Iota", "genre": ["Mystery"], "theme": ["Detective"], "studio": ["Bones"], "episodes": 26, "score": 8.0},
    {"title": "Kappa", "genre": ["Romance"], "episodes": 12, "score": 8.8},
]


def _world(rows=ROWS):
    snap = make_snapshot(rows)
    return snap, build_catalog(snap), build_engine(snap)


def _idx(snap, title: str) -> int:
    return list(snap.titles).index(title)


# ------------------------------------------------------------------------------------------ metrics


def test_metrics_match_hand_computed_values():
    ids = np.arange(1, 21)
    s = np.linspace(20, 1, 20)                                   # position p has rank p + 1
    m = query_metrics(s, ids, 0)
    assert (m["rank"], m["hit"], m["ndcg"], m["mrr"]) == (1, 1.0, 1.0, 1.0)
    m = query_metrics(s, ids, 7)
    assert m["rank"] == 8 and m["hit"] == 1.0 and m["ndcg"] == pytest.approx(1 / np.log2(9))
    m = query_metrics(s, ids, 8)
    assert m["rank"] == 9 and m["hit"] == 0.0 and m["ndcg"] == 0.0 and m["mrr"] == 0.0
    m = query_metrics(np.array([1.0, 1.0, 0.0]), np.array([1, 2, 3]), 0)
    assert m["auc"] == pytest.approx((1 + 0.5 * 1) / 2)          # a raw-score tie counts one half
    assert query_metrics(np.array([3.0]), np.array([1]), 0)["auc"] is None   # one candidate: undefined


def test_tie_break_is_deterministic_and_auc_ignores_ids():
    s = np.array([1.0, 1.0, 1.0, 0.5])
    a = query_metrics(s, np.array([20, 10, 30, 40]), 0)          # the target ties with ids 10 and 30
    b = query_metrics(s, np.array([5, 10, 30, 40]), 0)
    assert (a["rank"], b["rank"]) == (2, 1)                      # ties are broken by ascending id ...
    assert a["auc"] == b["auc"] == pytest.approx((1 + 0.5 * 2) / 3)   # ... but never count as discrimination
    assert rank_order(s, np.array([30, 20, 10, 40])).tolist() == [2, 1, 0, 3]
    scores = np.r_[s, np.zeros(40)]
    anime = np.r_[[30, 20, 10, 40], np.arange(100, 140)]
    assert top_k(scores, np.arange(44), anime, 3).tolist() == [2, 1, 0]
    assert top_k(scores, np.arange(44), anime, 3).tolist() == top_k(scores, np.arange(44), anime, 3).tolist()


def test_user_macro_and_paired_bootstrap_resample_users():
    users = np.array([1, 1, 2])
    assert user_macro(users, np.array([1.0, 0.0, 1.0])) == (pytest.approx(0.75), 2)   # users weigh equally
    r = paired_bootstrap({1: 1.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 1.0, 3: 0.0}, reps=500, seed=42)
    assert r["users"] == 3 and r["mean_diff"] == pytest.approx(1 / 3) and r["ci95"][0] <= r["ci95"][1]


# ------------------------------------------------------------------------------------------ rules


def test_keyword_and_normalisation_and_score_episode_boundaries():
    snap, cat, eng = _world()
    assert split_keywords(" MYSTERY ， comedy,mystery ") == ["mystery", "comedy"]   # full-width comma, case, dedupe
    m = eng.evaluate(ConstraintSet(require_recommendable=False, keywords=split_keywords("Mystery, COMEDY"))).mask
    assert [snap.titles[i] for i in np.flatnonzero(m)] == ["Theta Mystery Comedy"]  # AND over genre / title
    m = eng.evaluate(ConstraintSet(require_recommendable=False, require_scored=True, min_score=7.0,
                                   max_episodes=26)).mask
    names = {snap.titles[i] for i in np.flatnonzero(m)}
    assert {"Beta Comedy", "Iota"} <= names          # score 7.0 passes ">= 7.0"; 26 episodes pass "<= 26"
    assert "Delta" not in names                      # 6.99 < 7.0
    assert "Epsilon" not in names                    # 27 episodes
    assert "Gamma" not in names                      # unknown episode count fails an episode cap
    assert "Zeta Unscored" not in names              # no catalogue score: outside the algorithm-2 pool
    assert "Eta Explicit" not in names               # explicit content is never served
    assert set(np.flatnonzero(m)) <= set(cat.members.tolist())


def test_service_excludes_collection_never_pads_and_never_relaxes():
    snap, cat, eng = _world()
    svc = Alg2Service(cat, eng, {"A0": FormulaScorer("A0", 0.15)}, "A0", "ready", "test")
    H = np.array([_idx(snap, "Alpha Mystery")])
    res = svc.recommend(H, ConstraintSet(), "A0", 8)
    assert not set(res.items.tolist()) & set(H.tolist())                      # saved titles excluded
    assert set(res.candidates.tolist()) == set(cat.members.tolist()) - set(H.tolist())
    cs = ConstraintSet(keywords=["mystery"], min_score=7.0, max_episodes=26)
    res = svc.recommend(H, cs, "A0", 8)
    assert {snap.titles[i] for i in res.items} == {"Theta Mystery Comedy", "Iota"}   # 2 < 8: no padding
    cs = ConstraintSet(keywords=["no such word"], min_score=7.0)
    res = svc.recommend(H, cs, "A0", 8)
    assert len(res.items) == 0 and res.eligibility.count == 0
    assert cs.keywords == ["no such word"] and cs.min_score == 7.0                    # caller's filters untouched


# ------------------------------------------------------------------------------------------ profile


def test_profile_is_built_from_visible_history_only():
    rows_b = [dict(r) for r in ROWS]
    t = _idx(make_snapshot(ROWS), "Iota")
    rows_b[t] = {**rows_b[t], "genre": ["Comedy"], "theme": [], "studio": [], "episodes": 1, "score": 6.0}
    snap_a, cat_a, _ = _world(ROWS)
    snap_b, cat_b, _ = _world(rows_b)
    H = np.array([_idx(snap_a, "Alpha Mystery"), _idx(snap_a, "Theta Mystery Comedy")])
    pa, pb = build_profile(cat_a, H), build_profile(cat_b, H)
    assert pa.top_labels(cat_a) == pb.top_labels(cat_b) and pa.median_eps == pb.median_eps and pa.size == 2
    for ns in pa.freq:
        np.testing.assert_array_equal(pa.freq[ns], pb.freq[ns])
    others = np.array([i for i in cat_a.members.tolist() if i not in set(H.tolist()) | {t}])
    np.testing.assert_array_equal(compute_features(cat_a, pa, others), compute_features(cat_b, pb, others))


def test_features_follow_the_schema_and_mark_missing_values():
    snap, cat, _ = _world()
    prof = build_profile(cat, np.array([_idx(snap, "Alpha Mystery")]))
    cand = np.array([_idx(snap, "Iota"), _idx(snap, "Gamma"), _idx(snap, "Kappa")])
    F = compute_features(cat, prof, cand)
    assert F.shape == (3, len(feature_schema()["names"])) and F.dtype == np.float32
    assert F[0, 0] == pytest.approx(0.8) and F[0, 1] == 2           # Iota: score 8.0; Mystery + Detective hit
    assert np.isnan(F[1, 6])                                         # Gamma: unknown episodes -> length NaN
    assert F[2, 1] == 0 and F[2, 2] == 0.0                          # Kappa: no shared tag; genre known -> 0


# ------------------------------------------------------------------------------------------ bundles


def _bundle(tmp_path, cat, **changes):
    (tmp_path / "models").mkdir(exist_ok=True)
    model = tmp_path / "models" / "B1_seed42.txt"
    model.write_text("not a real model\n")
    import hashlib

    fz = {"catalog_hash": cat.catalog_hash, "feature_version": FEATURE_VERSION, "deploy_seed": 42,
          "formula": {"A0": {"hits_weight": 0.15}, "A1": {"hits_weight": 0.05}},
          "feature_schema": feature_schema(), "configs": {"B1": {"features": None}},
          "models": [{"family": "B1", "seed": 42, "file": "models/B1_seed42.txt",
                      "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}]}
    fz.update(changes)
    (tmp_path / "freeze_manifest.json").write_text(json.dumps(fz))
    return tmp_path


def test_incompatible_bundles_fail_loudly_and_the_service_falls_back(tmp_path):
    snap, cat, eng = _world()
    with pytest.raises(BundleMismatch, match="catalogue hash"):
        load_scorers(_bundle(tmp_path, cat, catalog_hash="0" * 64), cat)
    with pytest.raises(BundleMismatch, match="feature version"):
        load_scorers(_bundle(tmp_path, cat, feature_version="alg2-features-v0"), cat)
    run = _bundle(tmp_path, cat)
    (run / "models" / "B1_seed42.txt").write_text("changed after freeze\n")
    with pytest.raises(BundleMismatch, match="SHA-256"):
        load_scorers(run, cat)
    svc = build_service(snap, eng, {"scorer": "B1", "algorithm2_run": str(run)}, tmp_path)
    assert svc.status == "degraded" and svc.default == "A0" and "B1" not in svc.scorers
    label, scorer, fallback = svc.resolve("C0")
    assert label == "A0" and fallback == "C0 unavailable -> A0"
    ok = build_service(snap, eng, {"scorer": "A0", "algorithm2_run": None}, tmp_path)
    assert ok.status == "ready" and ok.default == "A0"
