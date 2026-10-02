"""Algorithm 2 as the website runs it: served ordering, rules, profile isolation, strict service behaviour
and bundle compatibility (app/recommendation/algorithm2). The experiment's own checks (metric definitions,
leakage, training, frozen-run reproducibility) live in algorithm2_benchmark/tests/.
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


# ------------------------------------------------------------------------------------------ ordering


def test_served_order_breaks_ties_by_ascending_id():
    s = np.array([1.0, 1.0, 1.0, 0.5])
    assert rank_order(s, np.array([30, 20, 10, 40])).tolist() == [2, 1, 0, 3]
    scores = np.r_[s, np.zeros(40)]
    anime = np.r_[[30, 20, 10, 40], np.arange(100, 140)]
    assert top_k(scores, np.arange(44), anime, 3).tolist() == [2, 1, 0]          # partial sort path, same order
    assert top_k(scores, np.arange(44), anime, 3).tolist() == top_k(scores, np.arange(44), anime, 3).tolist()


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
