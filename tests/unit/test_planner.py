import numpy as np
import pytest

from app.planning.duration import title_time
from app.planning.feasibility import PlanEntry, check
from app.planning.optimizer import PlanGA, PlanProblem, Unit, exhaustive_complete, greedy_complete
from app.planning.scheduling import build_schedule
from tests.helpers import make_snapshot


def _snap():
    return make_snapshot([
        {"title": "A", "episodes": 12, "duration_minutes": 24.0, "genre": ["Action"]},       # 288
        {"title": "B", "episodes": 24, "duration_minutes": 24.0, "genre": ["Drama"]},        # 576
        {"title": "C (movie)", "type": "Movie", "episodes": 1, "duration_minutes": 110.0, "genre": ["Romance"]},
        {"title": "D", "episodes": 13, "duration_minutes": 23.0, "genre": ["Comedy"]},       # 299
        {"title": "E", "episodes": None, "duration_minutes": 24.0, "genre": ["Action"]},     # unknown episodes
        {"title": "F sequel", "episodes": 12, "duration_minutes": 24.0, "genre": ["Action"]},
    ])


def _problem(snap, budget=900, max_titles=3, mode="complete", units=None, must=None, rel=None):
    idx = [0, 1, 2, 3]
    units = units or [Unit(i, [i]) for i in idx]
    items = sorted({i for u in units for i in u.items})
    times = {i: title_time(snap, i) for i in items}
    rng = np.random.default_rng(0)
    rel = rel or {i: float(v) for i, v in zip(items, rng.random(len(items)), strict=True)}
    sim = {(a, b): 0.3 for a in items for b in items if a < b}
    return PlanProblem([u.target for u in units], units, times, rel, {i: 0.5 for i in items}, None, sim, budget,
                       max_titles, 0, mode, set(must or []), {"relevance": 0.45, "quality": 0.2, "diversity": 0.2,
                                                               "utilization": 0.1})


def test_unknown_duration_is_not_plannable():
    snap = _snap()
    assert not title_time(snap, 4).known and title_time(snap, 4).full_cost() is None
    t = title_time(snap, 2)
    assert t.is_movie and t.full_cost() == 110.0


@pytest.mark.parametrize("budget", [150, 400, 700, 900, 1300])
def test_ga_matches_exhaustive_and_never_exceeds_budget(budget):
    snap = _snap()
    p = _problem(snap, budget=budget)
    ga = PlanGA(time_budget_ms=None, generations=80).solve(p, np.random.default_rng(1))
    oracle = exhaustive_complete(p)
    assert p.cost_items(ga.items) <= budget + 1e-9 and len(ga.items) <= 3
    assert ga.fitness >= oracle.fitness - 1e-6
    assert greedy_complete(p).fitness <= oracle.fitness + 1e-9


def test_prerequisite_units_are_planned_first_and_counted():
    snap = _snap()
    units = [Unit(0, [0]), Unit(5, [0, 5])]          # F requires A
    p = _problem(snap, budget=1000, units=units, rel={0: 0.2, 5: 0.9})
    sol = PlanGA(time_budget_ms=None).solve(p, np.random.default_rng(0))
    if 5 in sol.items:
        assert 0 in sol.items and sol.items.index(0) < sol.items.index(5)
        assert p.cost_items(sol.items) == pytest.approx(288 * 2)


def test_must_include_infeasible_is_reported_not_forced():
    snap = _snap()
    p = _problem(snap, budget=100, must=[1])        # B alone needs 576 min
    sol = PlanGA(time_budget_ms=None).solve(p, np.random.default_rng(0))
    assert sol.items == [] and sol.stopped == "infeasible"


def test_partial_mode_plans_next_episodes_within_budget():
    snap = _snap()
    p = _problem(snap, budget=200, max_titles=2, mode="partial")
    sol = PlanGA(time_budget_ms=None).solve(p, np.random.default_rng(0))
    assert p.feasible_items(sol.items, sol.episodes)
    for i in sol.items:
        t = p.times[i]
        assert 1 <= sol.episodes[i] <= (1 if t.is_movie else t.remaining)


def test_feasibility_checks():
    snap = _snap()
    times = {"a": title_time(snap, 0), "m": title_time(snap, 2), "f": title_time(snap, 5)}
    ok = check([PlanEntry(0, "a", 1, 12, 288.0, True)], times, 300, 3, 0, set(), "complete")
    assert ok.feasible and ok.remaining_minutes == pytest.approx(12)
    over = check([PlanEntry(0, "a", 1, 12, 288.0, True)], times, 200, 3, 0, set(), "complete")
    assert not over.feasible and "exceeds" in over.problems[0]
    split_movie = check([PlanEntry(2, "m", 1, 2, 110.0, True)], times, 900, 3, 0, set(), "complete")
    assert not split_movie.feasible
    wrong_order = check([PlanEntry(5, "f", 1, 12, 288.0, True, requires=["a"]), PlanEntry(0, "a", 1, 12, 288.0, True)],
                        times, 900, 3, 0, set(), "complete")
    assert any("must come first" in p for p in wrong_order.problems)
    missing_prereq = check([PlanEntry(5, "f", 1, 12, 288.0, True, requires=["a"])], times, 900, 3, 0, set(), "complete")
    assert any("neither watched nor in the plan" in p for p in missing_prereq.problems)
    watched_prereq = check([PlanEntry(5, "f", 1, 12, 288.0, True, requires=["a"])], times, 900, 3, 0, {"a"}, "complete")
    assert watched_prereq.feasible
    partial_in_complete = check([PlanEntry(0, "a", 1, 6, 144.0, False)], times, 900, 3, 0, set(), "complete")
    assert not partial_in_complete.feasible


def test_schedule_respects_capacity_and_never_splits_movies():
    snap = _snap()
    times = {"a": title_time(snap, 0), "m": title_time(snap, 2)}
    entries = [PlanEntry(0, "a", 1, 12, 288.0, True), PlanEntry(2, "m", 1, 1, 110.0, True)]
    # 398 min fit two 200-min sessions by total, but not by packing (24-min episodes + an unsplittable
    # 110-min movie): the scheduler must say so instead of splitting the movie
    packed = build_schedule(entries, times, {"a": "A", "m": "M"}, [{"label": "Sat", "minutes": 200},
                                                                  {"label": "Sun", "minutes": 200}])
    assert not packed["feasible"] and packed["unscheduled"]
    s = build_schedule(entries, times, {"a": "A", "m": "M"}, [{"label": "Sat", "minutes": 300},
                                                             {"label": "Sun", "minutes": 200}])
    assert s["feasible"]
    assert all(x["used_minutes"] <= x["capacity_minutes"] for x in s["sessions"])
    placed = sum(e["episode_end"] - e["episode_start"] + 1 for x in s["sessions"] for e in x["entries"] if e["item_key"] == "a")
    assert placed == 12
    tight = build_schedule(entries, times, {"a": "A", "m": "M"}, [{"label": "Sat", "minutes": 100}])
    assert not tight["feasible"] and tight["unscheduled"]
    assert any("longer than any single session" in p for p in tight["problems"])
