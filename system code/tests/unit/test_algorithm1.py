from pathlib import Path
import sys

import numpy as np
import pytest

# The experiment packages live next to this application directory (../Experiments/<name>), not inside it,
# so they are put on the import path here instead of being installed.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[3] / "Experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
ALGORITHM1_DIR = EXPERIMENTS_DIR / "algorithm1_benchmark"

from algorithm1_benchmark.kbbench.common import read_json, read_rows  # noqa: E402
from algorithm1_benchmark.kbbench.features import pair_features  # noqa: E402
from algorithm1_benchmark.kbbench.metrics import ranking_metrics  # noqa: E402
from app.core.config import load_settings
from app.recommendation.similarity.algorithm1 import Algorithm1Scorer
from tests.helpers import make_snapshot


@pytest.mark.parametrize("mode", ["real", "synthetic"])
def test_final_report_route_is_default_in_both_modes(mode):
    assert load_settings(mode, environ={}).recommendation["similar"]["scorer"] == "algorithm1_a"


def test_fixed_weights_missing_defaults_and_length_match_report():
    snap = make_snapshot([
        {"genre": ["Action", "Action"], "episodes": 12},
        {"genre": ["Action"], "episodes": 24},
        {"genre": [], "episodes": None},
        {"genre": [], "episodes": 0},
    ])
    result = Algorithm1Scorer().score(snap, 0, np.array([1, 2, 3]))
    assert result.sims["genre"].tolist() == [1, 0, 0]
    assert result.sims["theme"].tolist() == [0, 0, 0]
    assert result.sims["length"].tolist() == [0.5, 0.5, 0.5]
    assert result.score.tolist() == pytest.approx([0.4, 0.05, 0.05])
    assert result.coverage.tolist() == pytest.approx([0.45, 0, 0])
    assert not result.available["length"][1:].any()


def test_exact_scalar_projection_uses_primary_value_and_is_case_insensitive():
    snap = make_snapshot([
        {"studio": ["B", "A"], "demographic": ["Shounen"]},
        {"studio": ["a"], "demographic": ["shounen"]},
        {"studio": ["B"], "demographic": []},
    ])
    result = Algorithm1Scorer().score(snap, 0, np.array([1, 2]))
    assert result.sims["studio"].tolist() == [1, 0]
    assert result.sims["demographic"].tolist() == [1, 0]


def test_ties_use_numeric_anime_id_and_not_metadata_coverage():
    keys = np.array(["mal:100", "mal:20", "mal:3", "mal:1"])
    scores = np.array([0.5, 0.5, 0.5, 0.4])
    assert keys[Algorithm1Scorer.rank(scores, keys)].tolist() == ["mal:3", "mal:20", "mal:100", "mal:1"]


def test_runtime_route_matches_all_1000_published_targets():
    root = ALGORITHM1_DIR
    catalog = list(read_rows(root / "data/catalog.jsonl.gz"))
    snap = make_snapshot([
        {"item_key": f"mal:{r['anime_id']}", "title": r["title"], "genre": r["genres"], "theme": r["themes"],
         "demographic": [r["demographic"]] if r["demographic"] else [],
         "studio": [r["studio"]] if r["studio"] else [], "episodes": r["episodes"]}
        for r in catalog
    ])
    frozen = {r["target_id"]: r["items"] for r in read_rows(root / "results/synthetic_1000/predictions.jsonl")
              if r["method"] == "A"}
    queries = list(read_rows(root / "results/synthetic_1000/testset.jsonl"))
    assert len(catalog) == 14202 and len(queries) == len(frozen) == 1000
    scorer, metrics = Algorithm1Scorer(), []
    for query in queries:
        target = snap.index[f"mal:{query['target_id']}"]
        candidates = np.flatnonzero(np.arange(snap.n) != target)
        result = scorer.score(snap, target, candidates)
        order = scorer.rank(result.score, snap.keys[candidates])[:5]
        actual = [catalog[int(candidates[p])]["anime_id"] for p in order]
        expected = frozen[query["target_id"]]
        assert actual == [r["anime_id"] for r in expected], query["target_id"]
        for p, item in zip(order, expected, strict=True):
            candidate = int(candidates[p])
            assert result.score[p] == pytest.approx(item["similarity"], rel=0, abs=1e-12)
            features = pair_features(catalog[target], catalog[candidate])
            assert [result.sims[d][p] for d in scorer.weights] == pytest.approx(features, rel=0, abs=1e-12)
        metrics.append(ranking_metrics(actual, query["relevant_ids"]))
    summary = read_json(root / "results/synthetic_1000/summary.json")
    published = next(r for r in summary["summaries"] if r["method"] == "A")
    for metric in metrics[0]:
        assert float(np.mean([r[metric] for r in metrics])) == pytest.approx(published[metric], rel=0, abs=1e-12)
