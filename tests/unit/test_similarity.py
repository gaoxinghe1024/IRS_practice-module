import numpy as np

from app.recommendation.similarity.weighted import WeightedAttributeScorer
from tests.helpers import make_snapshot


def test_empty_sets_are_not_similar_and_weights_redistribute():
    snap = make_snapshot([
        {"genre": ["Action"], "theme": [], "episodes": 12},
        {"genre": ["Action"], "theme": [], "episodes": 12},
        {"genre": ["Drama"], "theme": ["School"], "episodes": 24},
    ])
    res = WeightedAttributeScorer().score(snap, 0, np.array([1, 2]))
    # theme empty on both sides -> unavailable, not 1.0
    assert not res.available["theme"][0] and np.isnan(res.sims["theme"][0])
    # remaining dims for item 1: genre (1.0) and length (1.0) -> 1.0 after redistribution
    assert np.isclose(res.score[0], 1.0)
    assert res.coverage[0] < 1.0


def test_insufficient_metadata_is_not_similar():
    snap = make_snapshot([{"genre": [], "episodes": None}, {"genre": ["Action"], "episodes": None}])
    res = WeightedAttributeScorer().score(snap, 0, np.array([1]))
    assert np.isnan(res.score[0]) and res.coverage[0] == 0


def test_jaccard_and_length_decay():
    snap = make_snapshot([
        {"genre": ["Action", "Comedy"], "studio": ["A"], "episodes": 12},
        {"genre": ["Action", "Drama"], "studio": ["A", "B"], "episodes": 24},
    ])
    s = WeightedAttributeScorer(tau=12.0)
    res = s.score(snap, 0, np.array([1]))
    assert np.isclose(res.sims["genre"][0], 1 / 3)
    assert np.isclose(res.sims["studio"][0], 1 / 2)          # multi-value studio kept as a set
    assert np.isclose(res.sims["length"][0], np.exp(-1.0))


def test_pairwise_matches_single_scores():
    snap = make_snapshot([{"genre": ["Action"], "episodes": 12}, {"genre": ["Action", "Drama"], "episodes": 13},
                          {"genre": ["Romance"], "episodes": None}])
    s = WeightedAttributeScorer()
    sim, valid = s.pairwise(snap, np.array([0, 1, 2]))
    single = s.score(snap, 0, np.array([1, 2]))
    assert np.isclose(sim[0, 1], single.score[0]) and np.isclose(sim[0, 2], single.score[1])
    assert not valid[0, 0]
