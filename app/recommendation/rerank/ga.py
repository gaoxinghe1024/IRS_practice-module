"""List optimisers over a fixed, already rule-filtered candidate pool.

relevance_greedy  top-K by normalised relevance (the GA's internal baseline and a seed)
mmr               maximal marginal relevance (separately named baseline, not the GA)
ga_list           genetic algorithm: chromosome = K distinct candidate positions;
                  tournament selection, set crossover that keeps shared genes and fills from
                  both parents, gene-replacement mutation, repair to K unique genes, elitism,
                  early stop and a wall-clock budget. The best-so-far individual is kept, so the
                  returned fitness is never below the greedy seed's fitness (this says nothing
                  about ranking accuracy).
exhaustive        enumerates every K-subset of a small pool (oracle for tests / optimality gap)

The GA optimises the list objective inside this pool only; it is not a global
catalogue optimum.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from app.recommendation.rerank.objective import ListObjective


@dataclass
class ListResult:
    optimizer: str
    selected: np.ndarray            # candidate positions, displayed order (relevance desc)
    fitness: float
    summary: dict
    log: list[dict] = field(default_factory=list)
    stopped: str = "completed"
    evaluations: int = 0
    note: str | None = None


def _display_order(obj: ListObjective, sel: np.ndarray, tiebreak: np.ndarray | None) -> np.ndarray:
    sel = np.asarray(sel)
    tb = tiebreak[sel] if tiebreak is not None else sel
    return sel[np.lexsort((tb, -obj.relevance[sel]))]


def relevance_greedy(obj: ListObjective, k: int, tiebreak: np.ndarray | None = None) -> ListResult:
    order = np.lexsort(((tiebreak if tiebreak is not None else np.arange(obj.size)), -obj.relevance))
    sel = order[:min(k, obj.size)]
    return ListResult("relevance_greedy", _display_order(obj, sel, tiebreak),
                      float(obj.fitness(sel[None, :])[0]), obj.summary(sel), evaluations=1)


def mmr(obj: ListObjective, k: int, lam: float = 0.7, tiebreak: np.ndarray | None = None) -> ListResult:
    chosen: list[int] = []
    remaining = list(range(obj.size))
    while remaining and len(chosen) < k:
        best, best_v = None, -np.inf
        for c in remaining:
            if chosen:
                sims = [obj.sim[c, j] for j in chosen if obj.valid[c, j]]
                red = max(sims) if sims else 0.0
            else:
                red = 0.0
            v = lam * obj.relevance[c] - (1 - lam) * red
            if v > best_v + 1e-12:
                best, best_v = c, v
        chosen.append(best)
        remaining.remove(best)
    sel = np.array(chosen)
    return ListResult(f"mmr_lambda_{lam}", _display_order(obj, sel, tiebreak),
                      float(obj.fitness(sel[None, :])[0]), obj.summary(sel), evaluations=len(chosen))


def exhaustive(obj: ListObjective, k: int, max_subsets: int = 200_000) -> ListResult:
    combos = list(itertools.islice(itertools.combinations(range(obj.size), min(k, obj.size)), max_subsets + 1))
    if len(combos) > max_subsets:
        raise ValueError("pool too large for exhaustive search")
    pop = np.array(combos)
    fit = obj.fitness(pop)
    b = int(np.argmax(fit))
    sel = pop[b]
    return ListResult("exhaustive_oracle", _display_order(obj, sel, None), float(fit[b]), obj.summary(sel),
                      evaluations=len(combos))


class GeneticListOptimizer:
    optimizer_id = "ga_list"
    version = "1.0"

    def __init__(self, population: int = 64, generations: int = 80, tournament_size: int = 3, elitism: int = 2,
                 crossover_prob: float = 0.8, mutation_prob: float = 0.2, early_stop_generations: int = 20,
                 time_budget_ms: float | None = 250, **_):
        self.population = int(population)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self.elitism = int(elitism)
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = float(mutation_prob)
        self.early_stop = int(early_stop_generations)
        self.time_budget_ms = time_budget_ms

    # ------------------------------------------------------------------ operators
    @staticmethod
    def _repair(child: list[int], k: int, n: int, rng: np.random.Generator) -> np.ndarray:
        seen: list[int] = []
        for g in child:
            if g not in seen and 0 <= g < n:
                seen.append(g)
        if len(seen) < k:
            pool = np.setdiff1d(np.arange(n), np.array(seen, dtype=int))
            seen.extend(rng.choice(pool, size=k - len(seen), replace=False).tolist())
        return np.array(seen[:k], dtype=int)

    def _crossover(self, a: np.ndarray, b: np.ndarray, k: int, n: int, rng) -> np.ndarray:
        common = np.intersect1d(a, b)
        others = np.setdiff1d(np.union1d(a, b), common)
        need = k - len(common)
        fill = rng.choice(others, size=min(need, len(others)), replace=False) if need > 0 and len(others) else []
        return self._repair(list(common) + list(fill), k, n, rng)

    def _mutate(self, child: np.ndarray, n: int, rng) -> np.ndarray:
        outside = np.setdiff1d(np.arange(n), child)
        if len(outside) == 0:
            return child
        c = child.copy()
        c[rng.integers(len(c))] = rng.choice(outside)
        return c

    def _tournament(self, fit: np.ndarray, rng) -> int:
        idx = rng.choice(len(fit), size=min(self.tournament_size, len(fit)), replace=False)
        return int(idx[np.argmax(fit[idx])])

    # ------------------------------------------------------------------ main loop
    def optimize(self, obj: ListObjective, k: int, rng: np.random.Generator,
                 seeds: list[np.ndarray] | None = None, tiebreak: np.ndarray | None = None) -> ListResult:
        n = obj.size
        if n <= k:
            sel = np.arange(n)
            return ListResult(self.optimizer_id, _display_order(obj, sel, tiebreak),
                              float(obj.fitness(sel[None, :])[0]) if n else 0.0,
                              obj.summary(sel) if n else {}, note="pool_not_larger_than_k", evaluations=1)
        t0 = time.perf_counter()
        greedy = relevance_greedy(obj, k, tiebreak).selected
        init = [np.sort(greedy)]
        for lam in (0.8, 0.6, 0.4):
            init.append(np.sort(mmr(obj, k, lam, tiebreak).selected))
        for s in seeds or []:
            init.append(np.sort(self._repair(list(s), k, n, rng)))
        while len(init) < self.population:
            init.append(np.sort(rng.choice(n, size=k, replace=False)))
        pop = np.array(init[:self.population])
        fit = obj.fitness(pop)
        evals = len(pop)
        b = int(np.argmax(fit))
        best, best_fit = pop[b].copy(), float(fit[b])
        log = [{"generation": 0, "best": round(best_fit, 5), "mean": round(float(fit.mean()), 5),
                "unique_genes": int(len(np.unique(pop)))}]
        stale, stopped = 0, "completed"
        for gen in range(1, self.generations + 1):
            order = np.argsort(-fit, kind="stable")
            nxt = [pop[i].copy() for i in order[:self.elitism]]
            while len(nxt) < self.population:
                a = pop[self._tournament(fit, rng)]
                bb = pop[self._tournament(fit, rng)]
                child = self._crossover(a, bb, k, n, rng) if rng.random() < self.crossover_prob else a.copy()
                if rng.random() < self.mutation_prob:
                    child = self._mutate(child, n, rng)
                nxt.append(np.sort(self._repair(list(child), k, n, rng)))
            pop = np.array(nxt)
            fit = obj.fitness(pop)
            evals += len(pop)
            g = int(np.argmax(fit))
            if fit[g] > best_fit + 1e-12:
                best, best_fit, stale = pop[g].copy(), float(fit[g]), 0
            else:
                stale += 1
            log.append({"generation": gen, "best": round(best_fit, 5), "mean": round(float(fit.mean()), 5),
                        "unique_genes": int(len(np.unique(pop)))})
            if stale >= self.early_stop:
                stopped = "early_stop"
                break
            if self.time_budget_ms and (time.perf_counter() - t0) * 1000 > self.time_budget_ms:
                stopped = "time_budget"
                break
        assert len(set(best.tolist())) == k   # final feasibility re-check: K unique genes
        return ListResult(self.optimizer_id, _display_order(obj, best, tiebreak), best_fit, obj.summary(best),
                          log=log, stopped=stopped, evaluations=evals)
