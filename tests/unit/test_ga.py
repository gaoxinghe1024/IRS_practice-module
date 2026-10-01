import numpy as np
import pytest

from app.recommendation.rerank.ga import GeneticListOptimizer, exhaustive, mmr, relevance_greedy
from app.recommendation.rerank.objective import ListObjective, fresh_flags, novelty_scores


def _objective(n=11, seed=0, weights=None, novelty=True, fresh=True):
    rng = np.random.default_rng(seed)
    rel = rng.random(n)
    sim = rng.random((n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)
    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)
    return ListObjective(rel, sim, valid, rng.random(n) if novelty else None,
                         rng.random(n) > 0.5 if fresh else None,
                         weights or {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2, "fresh": 0.1})


@pytest.mark.parametrize("seed", range(6))
def test_ga_reaches_exhaustive_optimum_on_small_pools(seed):
    obj = _objective(n=11, seed=seed)
    oracle = exhaustive(obj, 5)
    ga = GeneticListOptimizer(population=64, generations=120, time_budget_ms=None).optimize(
        obj, 5, np.random.default_rng(seed))
    assert ga.fitness <= oracle.fitness + 1e-9
    assert oracle.fitness - ga.fitness < 1e-6       # optimality gap on a 462-subset pool


def test_ga_never_below_greedy_seed_and_returns_unique_items():
    for seed in range(10):
        obj = _objective(n=20, seed=seed)
        greedy = relevance_greedy(obj, 10)
        ga = GeneticListOptimizer(time_budget_ms=None).optimize(obj, 10, np.random.default_rng(seed))
        assert ga.fitness >= greedy.fitness - 1e-12
        assert len(set(ga.selected.tolist())) == 10
        assert ga.log[0]["generation"] == 0 and len(ga.log) > 1


def test_ga_is_deterministic_for_a_seed():
    obj = _objective(n=20, seed=4)
    a = GeneticListOptimizer(time_budget_ms=None).optimize(obj, 10, np.random.default_rng(42))
    b = GeneticListOptimizer(time_budget_ms=None).optimize(obj, 10, np.random.default_rng(42))
    assert a.selected.tolist() == b.selected.tolist() and a.fitness == b.fitness


def test_crossover_combines_parents_and_mutation_changes_a_gene():
    ga = GeneticListOptimizer()
    rng = np.random.default_rng(0)
    a, b = np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5])
    child = ga._crossover(a, b, 4, 10, rng)
    assert {2, 3} <= set(child.tolist())                 # shared genes kept
    assert set(child.tolist()) <= {0, 1, 2, 3, 4, 5}      # only parental genes
    m = ga._mutate(np.array([0, 1, 2, 3]), 10, rng)
    assert len(set(m.tolist()) - {0, 1, 2, 3}) == 1


def test_small_pool_returns_all_candidates():
    obj = _objective(n=4, seed=1)
    res = GeneticListOptimizer().optimize(obj, 10, np.random.default_rng(0))
    assert sorted(res.selected.tolist()) == [0, 1, 2, 3] and res.note == "pool_not_larger_than_k"


def test_single_item_diversity_unavailable_and_weights_redistributed():
    obj = _objective(n=5, seed=2, novelty=False, fresh=False)
    t = obj.terms(np.array([[3]]))
    assert np.isnan(t["diversity"][0]) and np.isnan(t["novelty"][0]) and np.isnan(t["fresh"][0])
    # only relevance available -> fitness equals relevance
    assert np.isclose(obj.fitness(np.array([[3]]))[0], obj.relevance[3])


def test_novelty_unavailable_when_all_counts_zero_and_fresh_needs_history():
    assert novelty_scores(np.zeros(5), np.arange(5)) is None
    nov = novelty_scores(np.array([0, 9, 99, 999]), np.arange(4))
    assert nov[0] == 1.0 and nov[-1] == 0.0 and np.all(np.diff(nov) < 0)
    assert fresh_flags(None, np.arange(3), set()) is None


def test_mmr_is_distinct_from_ga_and_greedy_ids():
    obj = _objective(n=12, seed=3)
    assert mmr(obj, 5).optimizer.startswith("mmr")
    assert relevance_greedy(obj, 5).optimizer == "relevance_greedy"
