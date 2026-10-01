import numpy as np

from app.recommendation.constraints import RULES, ConstraintEngine, ConstraintSet
from tests.helpers import make_snapshot


def _snap():
    return make_snapshot([
        {"title": "Alpha Mystery", "genre": ["Mystery"], "episodes": 12, "score": 8.5},
        {"title": "Beta Comedy", "genre": ["Comedy"], "episodes": 24, "score": 7.0},
        {"title": "Gamma", "genre": ["Mystery", "Horror"], "episodes": None, "score": 9.0},
        {"title": "Delta", "genre": ["Comedy"], "episodes": 6, "score": None, "is_recommendable": False},
        {"title": "Epsilon", "genre": ["Mystery"], "episodes": 10, "score": 6.0, "is_explicit": True,
         "in_catalog": False},
    ])


def test_missing_values_never_satisfy_hard_filters():
    eng = ConstraintEngine(_snap())
    el = eng.evaluate(ConstraintSet(require_recommendable=False, max_episodes=12))
    assert not el.mask[2]                       # unknown episodes fails "<= 12"
    el = eng.evaluate(ConstraintSet(require_recommendable=False, min_score=6.5))
    assert not el.mask[3]                       # unknown score fails ">= 6.5"
    assert not el.mask[4]                       # explicit content always excluded


def test_keywords_all_must_match_genre_or_title():
    eng = ConstraintEngine(_snap())
    el = eng.evaluate(ConstraintSet(keywords=["myst", "alpha"]))
    assert el.mask.tolist() == [True, False, False, False, False]
    el = eng.evaluate(ConstraintSet(keywords=["myst"]))
    assert el.mask.tolist() == [True, False, True, False, False]


def test_exclusion_of_unknown_namespace_fails_closed():
    snap = make_snapshot([{"genre": ["Action"], "theme": ["School"]}, {"genre": ["Action"], "theme": []}])
    el = ConstraintEngine(snap).evaluate(ConstraintSet(exclude_tags=[("theme", "Gore")]))
    assert el.mask.tolist() == [True, False]   # themes unknown -> cannot verify "no Gore"
    assert el.unknown_tag_namespaces == {"theme": 1}


def test_relaxation_is_explicit_and_never_touches_protected_rules():
    eng = ConstraintEngine(_snap())
    cs = ConstraintSet(max_episodes=5, min_score=8.0, exclude_tags=[("genre", "Horror")])
    assert eng.evaluate(cs).count == 0
    opts = eng.suggest_relaxations(cs, k=1)
    assert opts, "a relaxation option is offered"
    for o in opts:
        assert RULES.get(o["rule_id"], {"relaxable": True})["relaxable"] or o["rule_id"] == "filter.*"
        assert "exclude_tags" not in o["change"]
    # applying the suggested episode cap yields the promised candidates
    ep = next(o for o in opts if o["rule_id"] == "filter.max_episodes")
    assert eng.evaluate(cs.copy(max_episodes=ep["change"]["max_episodes"])).count == ep["candidate_count"]


def test_clear_filters_option_when_no_single_change_helps():
    eng = ConstraintEngine(_snap())
    cs = ConstraintSet(keywords=["zzzz"], min_score=9.9, max_episodes=1)
    opts = eng.suggest_relaxations(cs, k=3)
    assert opts and opts[0]["rule_id"] == "filter.*"


def test_fail_reasons_are_reported_per_rule():
    eng = ConstraintEngine(_snap())
    el = eng.evaluate(ConstraintSet(max_episodes=12, min_score=8.0))
    assert set(el.fail_reasons(1)) == {"filter.max_episodes", "filter.min_score"}
    assert el.fail_counts["content.explicit"] == 1
    assert np.array_equal(el.mask, el.rule_masks["content.explicit"] & el.rule_masks["pool.recommendable"]
                          & el.rule_masks["filter.max_episodes"] & el.rule_masks["filter.min_score"])
