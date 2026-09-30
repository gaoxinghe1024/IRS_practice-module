"""Viewing-plan optimisation under a time budget (a constrained knapsack-style problem).

complete mode  unit = a title plus its unmet confirmed prerequisites (watched first);
               chromosome = one bit per unit. The plan is the union of selected units' titles.
partial mode   only when the user asks for it: gene = number of next episodes to watch per title
               (from the episode after their progress). Titles needing an unwatched prequel are
               excluded in this mode.

    F = sum_t w_t * term_t / sum_t w_t over available terms
        relevance / quality  minute-weighted means over planned titles
        diversity            mean pairwise (1 - content similarity), unavailable for < 2 titles
        novelty              mean train-popularity novelty
        utilisation          min(1, planned minutes / budget) with a small weight, so the plan is
                             not padded with low-relevance titles just to fill time

Budget, title cap, prerequisites and must-include titles are hard: every chromosome is
repaired before evaluation and the final plan is re-validated (feasibility.check).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_WEIGHTS = {"relevance": 0.45, "quality": 0.2, "diversity": 0.2, "novelty": 0.05, "utilization": 0.1}


@dataclass
class Unit:
    target: int
    items: list[int]          # watch order: prerequisites first, then the target


@dataclass
class PlanProblem:
    candidates: list[int]
    units: list[Unit]
    times: dict                       # idx -> TitleTime
    relevance: dict[int, float]
    quality: dict[int, float]
    novelty: dict[int, float] | None
    sim: dict[tuple[int, int], float]
    budget: float
    max_titles: int
    break_minutes: int
    mode: str
    must: set[int]
    weights: dict[str, float]

    # ---------------------------------------------------------------- complete mode
    def decode_bits(self, bits: np.ndarray) -> list[int]:
        order: list[int] = []
        for u, on in enumerate(bits):
            if on:
                for i in self.units[u].items:
                    if i not in order:
                        order.append(i)
        return order

    def cost_items(self, items: list[int], episodes: dict[int, int] | None = None) -> float:
        total = 0.0
        for i in items:
            t = self.times[i]
            c = t.full_cost() if episodes is None else t.partial_cost(episodes[i])
            total += (c or 0.0) + self.break_minutes
        return total

    def minutes(self, items: list[int], episodes: dict[int, int] | None = None) -> dict[int, float]:
        return {i: (self.times[i].full_cost() if episodes is None else self.times[i].partial_cost(episodes[i])) or 0.0
                for i in items}

    def fitness_items(self, items: list[int], episodes: dict[int, int] | None = None) -> tuple[float, dict]:
        if not items:
            return 0.0, {"relevance": None, "quality": None, "diversity": None, "novelty": None, "utilization": 0.0}
        m = self.minutes(items, episodes)
        tot = sum(m.values()) or 1.0
        terms = {"relevance": sum(m[i] * self.relevance.get(i, 0.0) for i in items) / tot,
                 "quality": sum(m[i] * self.quality.get(i, 0.0) for i in items) / tot,
                 "utilization": min(1.0, self.cost_items(items, episodes) / self.budget)}
        pairs = [self.sim.get((a, b), self.sim.get((b, a))) for a, b in itertools.combinations(items, 2)]
        pairs = [p for p in pairs if p is not None]
        terms["diversity"] = float(np.mean([1 - p for p in pairs])) if pairs else None
        terms["novelty"] = float(np.mean([self.novelty[i] for i in items])) if self.novelty else None
        num = den = 0.0
        for k, w in self.weights.items():
            if terms.get(k) is not None:
                num += w * terms[k]
                den += w
        return (num / den if den else 0.0), terms

    def feasible_items(self, items: list[int], episodes: dict[int, int] | None = None) -> bool:
        return self.cost_items(items, episodes) <= self.budget + 1e-9 and len(items) <= self.max_titles

    def unit_value(self, u: int) -> float:
        t = self.units[u].target
        return self.relevance.get(t, 0.0) * 0.7 + self.quality.get(t, 0.0) * 0.3

    def repair_bits(self, bits: np.ndarray) -> np.ndarray | None:
        bits = bits.copy()
        must_units = [u for u, un in enumerate(self.units) if un.target in self.must]
        bits[must_units] = 1
        while True:
            items = self.decode_bits(bits)
            if self.feasible_items(items):
                return bits
            on = [u for u in np.flatnonzero(bits) if u not in must_units]
            if not on:
                return None
            base = self.cost_items(items)

            def density(u):
                b2 = bits.copy()
                b2[u] = 0
                saved = base - self.cost_items(self.decode_bits(b2))
                return self.unit_value(u) / max(1.0, saved)

            worst = min(on, key=lambda u: (density(u), u))
            bits[worst] = 0

    def improve_bits(self, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One pass of fitness-improving additions (never adds a title that lowers F)."""
        fit, _ = self.fitness_items(self.decode_bits(bits))
        for u in rng.permutation(len(self.units)):
            if bits[u]:
                continue
            b2 = bits.copy()
            b2[u] = 1
            items = self.decode_bits(b2)
            if self.feasible_items(items):
                f2, _ = self.fitness_items(items)
                if f2 > fit + 1e-12:
                    bits, fit = b2, f2
        return bits

    # ---------------------------------------------------------------- partial mode
    def decode_genes(self, genes: np.ndarray) -> tuple[list[int], dict[int, int]]:
        items = [self.candidates[k] for k in np.flatnonzero(genes)]
        return items, {self.candidates[k]: int(genes[k]) for k in np.flatnonzero(genes)}

    def gene_max(self, k: int) -> int:
        t = self.times[self.candidates[k]]
        return 1 if t.is_movie else int(t.remaining or 0)

    def repair_genes(self, genes: np.ndarray) -> np.ndarray | None:
        genes = genes.copy()
        for k, i in enumerate(self.candidates):
            genes[k] = int(np.clip(genes[k], 0, self.gene_max(k)))
            if i in self.must and genes[k] == 0:
                genes[k] = self.gene_max(k)
        while True:
            items, eps = self.decode_genes(genes)
            if self.feasible_items(items, eps):
                return genes
            on = [k for k in np.flatnonzero(genes)]
            if not on:
                return None
            if len(items) > self.max_titles:
                drop = min((k for k in on if self.candidates[k] not in self.must),
                           key=lambda k: (self.relevance.get(self.candidates[k], 0), k), default=None)
                if drop is None:
                    return None
                genes[drop] = 0
                continue
            k = min(on, key=lambda k: (self.relevance.get(self.candidates[k], 0), -genes[k], k))
            if self.candidates[k] in self.must and genes[k] <= 1:
                others = [x for x in on if self.candidates[x] not in self.must]
                if not others:
                    return None
                k = min(others, key=lambda k: (self.relevance.get(self.candidates[k], 0), k))
            genes[k] -= 1


@dataclass
class PlanSolution:
    optimizer: str
    items: list[int]
    episodes: dict[int, int] | None
    fitness: float
    terms: dict
    log: list[dict] = field(default_factory=list)
    stopped: str = "completed"
    evaluations: int = 0


def greedy_complete(p: PlanProblem) -> PlanSolution:
    bits = np.zeros(len(p.units), dtype=np.int8)
    fixed = p.repair_bits(bits)
    if fixed is None:
        return PlanSolution("greedy_density", [], None, 0.0, p.fitness_items([])[1])
    bits = fixed
    order = sorted(range(len(p.units)), key=lambda u: (-p.unit_value(u) / max(1.0, p.cost_items(p.units[u].items)), u))
    for u in order:
        b2 = bits.copy()
        b2[u] = 1
        if p.feasible_items(p.decode_bits(b2)):
            bits = b2
    items = p.decode_bits(bits)
    f, terms = p.fitness_items(items)
    return PlanSolution("greedy_density", items, None, f, terms, evaluations=len(order))


def exhaustive_complete(p: PlanProblem, max_units: int = 14) -> PlanSolution:
    if len(p.units) > max_units:
        raise ValueError("too many units for the exact oracle")
    best, best_f, best_terms, evals = [], -1.0, {}, 0
    for r in range(len(p.units) + 1):
        for combo in itertools.combinations(range(len(p.units)), r):
            bits = np.zeros(len(p.units), dtype=np.int8)
            bits[list(combo)] = 1
            if any(p.units[u].target in p.must for u in range(len(p.units)) if not bits[u]):
                continue
            items = p.decode_bits(bits)
            evals += 1
            if not p.feasible_items(items):
                continue
            f, terms = p.fitness_items(items)
            if f > best_f + 1e-12:
                best, best_f, best_terms = items, f, terms
    return PlanSolution("exhaustive_oracle", best, None, max(best_f, 0.0), best_terms, evaluations=evals)


class PlanGA:
    optimizer_id = "plan_ga"
    version = "1.0"

    def __init__(self, population: int = 60, generations: int = 120, tournament_size: int = 3, elitism: int = 2,
                 crossover_prob: float = 0.8, mutation_prob: float = 0.3, early_stop_generations: int = 25,
                 time_budget_ms: float | None = 1500, **_):
        self.population, self.generations = int(population), int(generations)
        self.tournament_size, self.elitism = int(tournament_size), int(elitism)
        self.crossover_prob, self.mutation_prob = float(crossover_prob), float(mutation_prob)
        self.early_stop, self.time_budget_ms = int(early_stop_generations), time_budget_ms

    def solve(self, p: PlanProblem, rng: np.random.Generator) -> PlanSolution:
        return self._complete(p, rng) if p.mode == "complete" else self._partial(p, rng)

    def _tournament(self, fit, rng) -> int:
        idx = rng.choice(len(fit), size=min(self.tournament_size, len(fit)), replace=False)
        return int(idx[np.argmax(fit[idx])])

    def _complete(self, p: PlanProblem, rng) -> PlanSolution:
        t0 = time.perf_counter()
        n = len(p.units)
        if n == 0 or p.repair_bits(np.zeros(n, dtype=np.int8)) is None:
            # nothing to plan, or the must-include titles alone do not fit
            return PlanSolution(self.optimizer_id, [], None, 0.0, p.fitness_items([])[1],
                                stopped="empty" if n == 0 else "infeasible")
        greedy = greedy_complete(p)
        seed_bits = np.array([1 if set(u.items) <= set(greedy.items) and u.target in greedy.items else 0
                              for u in p.units], dtype=np.int8)
        pop = [p.repair_bits(seed_bits)]
        attempts = 0
        while len(pop) < self.population and attempts < self.population * 20:
            attempts += 1
            b = (rng.random(n) < min(0.5, 3.0 / max(1, n))).astype(np.int8)
            r = p.repair_bits(b)
            if r is not None:
                pop.append(r)
        pop = np.array(pop)

        def evaluate(pop):
            return np.array([p.fitness_items(p.decode_bits(b))[0] for b in pop])

        fit = evaluate(pop)
        evals = len(pop)
        b = int(np.argmax(fit))
        best, best_f = pop[b].copy(), float(fit[b])
        log = [{"generation": 0, "best": round(best_f, 5), "mean": round(float(fit.mean()), 5)}]
        stale, stopped = 0, "completed"
        for gen in range(1, self.generations + 1):
            order = np.argsort(-fit, kind="stable")
            nxt = [pop[i].copy() for i in order[:self.elitism]]
            while len(nxt) < self.population:
                a, c = pop[self._tournament(fit, rng)], pop[self._tournament(fit, rng)]
                child = np.where(rng.random(n) < 0.5, a, c) if rng.random() < self.crossover_prob else a.copy()
                if rng.random() < self.mutation_prob:
                    child[rng.integers(n)] ^= 1
                child = p.repair_bits(child)
                if child is None:
                    continue
                if rng.random() < 0.3:
                    child = p.improve_bits(child, rng)
                nxt.append(child)
            pop = np.array(nxt)
            fit = evaluate(pop)
            evals += len(pop)
            g = int(np.argmax(fit))
            if fit[g] > best_f + 1e-12:
                best, best_f, stale = pop[g].copy(), float(fit[g]), 0
            else:
                stale += 1
            log.append({"generation": gen, "best": round(best_f, 5), "mean": round(float(fit.mean()), 5)})
            if stale >= self.early_stop:
                stopped = "early_stop"
                break
            if self.time_budget_ms and (time.perf_counter() - t0) * 1000 > self.time_budget_ms:
                stopped = "time_budget"
                break
        items = p.decode_bits(best)
        f, terms = p.fitness_items(items)
        return PlanSolution(self.optimizer_id, items, None, f, terms, log, stopped, evals)

    def _partial(self, p: PlanProblem, rng) -> PlanSolution:
        t0 = time.perf_counter()
        n = len(p.candidates)
        if n == 0:
            return PlanSolution(self.optimizer_id, [], {}, 0.0, p.fitness_items([])[1])
        maxg = np.array([p.gene_max(k) for k in range(n)])
        # greedy seed: whole titles by relevance while they fit, then fill the rest with next episodes
        seed = np.zeros(n, dtype=np.int64)
        for k in sorted(range(n), key=lambda k: (-p.relevance.get(p.candidates[k], 0), k)):
            trial = seed.copy()
            trial[k] = maxg[k]
            items, eps = p.decode_genes(trial)
            if p.feasible_items(items, eps):
                seed = trial
        first = p.repair_genes(seed)
        if first is None:
            return PlanSolution(self.optimizer_id, [], {}, 0.0, p.fitness_items([])[1], stopped="infeasible")
        pop = [first]
        attempts = 0
        while len(pop) < self.population and attempts < self.population * 20:
            attempts += 1
            g = np.where(rng.random(n) < min(0.5, 3.0 / max(1, n)), rng.integers(0, maxg + 1), 0)
            r = p.repair_genes(g)
            if r is not None:
                pop.append(r)
        pop = np.array(pop)

        def evaluate(pop):
            return np.array([p.fitness_items(*p.decode_genes(g))[0] for g in pop])

        fit = evaluate(pop)
        evals = len(pop)
        b = int(np.argmax(fit))
        best, best_f = pop[b].copy(), float(fit[b])
        log = [{"generation": 0, "best": round(best_f, 5), "mean": round(float(fit.mean()), 5)}]
        stale, stopped = 0, "completed"
        for gen in range(1, self.generations + 1):
            order = np.argsort(-fit, kind="stable")
            nxt = [pop[i].copy() for i in order[:self.elitism]]
            while len(nxt) < self.population:
                a, c = pop[self._tournament(fit, rng)], pop[self._tournament(fit, rng)]
                child = np.where(rng.random(n) < 0.5, a, c) if rng.random() < self.crossover_prob else a.copy()
                if rng.random() < self.mutation_prob:
                    k = rng.integers(n)
                    child[k] = int(np.clip(child[k] + rng.choice([-2, -1, 1, 2]), 0, maxg[k]))
                child = p.repair_genes(child)
                if child is not None:
                    nxt.append(child)
            pop = np.array(nxt)
            fit = evaluate(pop)
            evals += len(pop)
            g = int(np.argmax(fit))
            if fit[g] > best_f + 1e-12:
                best, best_f, stale = pop[g].copy(), float(fit[g]), 0
            else:
                stale += 1
            log.append({"generation": gen, "best": round(best_f, 5), "mean": round(float(fit.mean()), 5)})
            if stale >= self.early_stop:
                stopped = "early_stop"
                break
            if self.time_budget_ms and (time.perf_counter() - t0) * 1000 > self.time_budget_ms:
                stopped = "time_budget"
                break
        items, eps = p.decode_genes(best)
        f, terms = p.fitness_items(items, eps)
        return PlanSolution(self.optimizer_id, items, eps, f, terms, log, stopped, evals)
