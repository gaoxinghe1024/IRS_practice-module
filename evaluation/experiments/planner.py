"""Viewing-plan evaluation: greedy density vs GA vs exact oracle under hard feasibility.

Offline users (ranking-train histories) get plan requests from a fixed grid of budgets, title caps and
variety settings. Every returned plan is re-validated independently (budget, title cap, prerequisites,
episode ranges); any infeasible plan is a failure. On small candidate pools the exhaustive oracle gives
the true optimum of the same objective, so the GA's optimality gap is measured, not assumed.
"""

from __future__ import annotations

import time

import numpy as np

from app.planning.optimizer import PlanGA, exhaustive_complete, greedy_complete
from app.recommendation.profiles import UserState
from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci
from evaluation.protocols import load_context, sample_users
from evaluation.sandbox import sandbox_settings
from pipelines.split import load_split

GRID = [(300, 2, "normal"), (600, 3, "normal"), (900, 3, "high"), (1500, 5, "normal"), (900, 5, "low")]


def _eval_solution(problem, sol):
    items = sol.items
    feasible = problem.feasible_items(items, sol.episodes)
    cost = problem.cost_items(items, sol.episodes) if items else 0.0
    return {"fitness": sol.fitness, "feasible": float(feasible), "utilization": cost / problem.budget,
            "titles": len(items), "relevance": (sol.terms or {}).get("relevance") or 0.0,
            "diversity": (sol.terms or {}).get("diversity")}


def run(settings, sample_users_n: int | None = None, seeds: int = 3, tag=None, **_) -> dict:
    from app.db.session import session_scope
    from app.services import plans
    from app.services.serving import load_state

    ctx = load_context(settings, load_models=False)
    n = sample_users_n or (120 if settings.mode == "real" else 60)
    run = Run(settings, "planner", {"sample_users": n, "grid": GRID, "seeds": seeds, "tag": tag})
    test = load_split(settings, "ranking", "test", ctx.version)
    users = sample_users(test, n, seed=53)
    hists = ctx.histories(users)
    res: dict[str, list[dict]] = {}
    gaps = []
    with sandbox_settings(settings) as sb:
        state = load_state(sb)
        for u in users:
            if u not in hists:
                continue
            items, r = hists[u]
            us = UserState(user_id=None, profile_version=0,
                           ratings={int(i): float(x) for i, x in zip(items, r, strict=True)})
            for budget, max_titles, div in GRID:
                req = {"budget_minutes": budget, "max_titles": max_titles, "mode": "complete", "diversity": div,
                       "include_genres": [], "exclude_genres": [], "min_score": None, "max_episodes_per_title": None,
                       "allow_prerequisites": True, "break_minutes_per_title": 0, "daily_slots": [], "must_include": []}
                with session_scope(sb.db_url) as db:
                    problem, _ctx = plans.build_problem(state, db, None, req, us=us)
                key = f"b{budget}_t{max_titles}_{div}"
                t0 = time.perf_counter()
                g = greedy_complete(problem)
                tg = (time.perf_counter() - t0) * 1000
                res.setdefault(f"{key}:greedy", []).append({**_eval_solution(problem, g), "ms": tg})
                for sd in range(seeds):
                    t0 = time.perf_counter()
                    sol = PlanGA(time_budget_ms=None, generations=120).solve(problem, np.random.default_rng(sd))
                    ms = (time.perf_counter() - t0) * 1000
                    m = {**_eval_solution(problem, sol), "ms": ms}
                    res.setdefault(f"{key}:ga_s{sd}", []).append(m)
                    if not m["feasible"]:
                        run.failure({"user": u, "request": key, "reason": "GA returned an infeasible plan"})
                # oracle on a reduced pool (first 12 units by relevance) for the optimality gap
                small = plans.PlanProblem(**{**problem.__dict__, "units": problem.units[:12],
                                             "candidates": problem.candidates[:12]})
                if small.units:
                    ora = exhaustive_complete(small)
                    ga_small = PlanGA(time_budget_ms=None, generations=120).solve(small, np.random.default_rng(0))
                    gaps.append(ora.fitness - ga_small.fitness)
                run.case({"user": u, "request": key, "greedy_fitness": round(g.fitness, 5),
                          "ga_fitness": round(res[f"{key}:ga_s0"][-1]["fitness"], 5),
                          "ga_utilization": round(res[f"{key}:ga_s0"][-1]["utilization"], 4)})
    rows, metrics = [], {"users": len(hists), "results": {}}
    for key, vals in sorted(res.items()):
        agg = {k: bootstrap_ci([v[k] for v in vals if v[k] is not None]) for k in vals[0]}
        metrics["results"][key] = agg
        req, method = key.split(":")
        if method in ("greedy", "ga_s0"):
            rows.append({"request": req, "method": "ga" if method == "ga_s0" else method,
                         **{k: agg[k]["mean"] for k in ("fitness", "feasible", "utilization", "titles", "relevance",
                                                        "diversity", "ms")}, "n": agg["fitness"]["n"]})
    metrics["oracle_small_pool"] = {"cases": len(gaps), "mean_gap": float(np.mean(gaps)) if gaps else None,
                                    "max_gap": float(np.max(gaps)) if gaps else None,
                                    "share_optimal": float(np.mean(np.array(gaps) < 1e-9)) if gaps else None}
    seed_spread = {}
    for budget, max_titles, div in GRID:
        key = f"b{budget}_t{max_titles}_{div}"
        seed_spread[key] = [round(float(np.mean([v["fitness"] for v in res.get(f"{key}:ga_s{s}", [])])), 5)
                            for s in range(seeds)]
    metrics["ga_seed_mean_fitness"] = seed_spread
    infeasible = sum(1 for k, v in res.items() for x in v if not x["feasible"])
    metrics["infeasible_plans"] = infeasible
    run.comparison(rows)
    lines = [f"# Viewing-plan optimisation - {settings.mode} data", "",
             f"{metrics['users']} offline users (train histories as profiles), {len(GRID)} request types, complete-title "
             "mode, prerequisites allowed. Every plan was re-validated independently.", ""]
    if settings.mode == "synthetic":
        lines += ["> Synthetic data: validates optimiser correctness and feasibility, not plan quality.", ""]
    lines += [md_table(rows, ["request", "method", "fitness", "feasible", "utilization", "titles", "relevance",
                              "diversity", "ms", "n"]), "",
              f"Infeasible plans returned: {infeasible}.", "",
              f"Exact oracle on 12-unit pools: {metrics['oracle_small_pool']}.", "",
              f"GA mean fitness by seed: {seed_spread}.", "",
              "Utilisation has a small weight in the objective on purpose: the plan should not be padded with "
              "low-relevance titles only to fill the budget."]
    return run.finish(metrics, "\n".join(lines) + "\n", {"grid": GRID})
