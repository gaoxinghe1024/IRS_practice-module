"""Viewing plans: compute (preview), save, edit (re-validated), re-optimise, export.

Plans reuse the recommendation signals (ICF / content / popularity relevance,
attribute similarity, novelty) but have their own feasibility rules. Every
stored revision carries the request, the per-item duration evidence, the
algorithm id/version, the dataset version and a freshly computed feasibility.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFound, ValidationFailed
from app.db.models import PlanItem, PlanRevision, PlanSession, User, ViewingPlan, utcnow
from app.graph.relations import PrequelGraph, confirmed_edges
from app.planning import export as plan_export
from app.planning.duration import TitleTime, title_time
from app.planning.feasibility import PlanEntry, check
from app.planning.optimizer import DEFAULT_WEIGHTS, PlanGA, PlanProblem, Unit, exhaustive_complete, greedy_complete
from app.planning.scheduling import build_schedule
from app.recommendation import explanations as X
from app.recommendation.constraints import ConstraintSet
from app.recommendation.features import minmax
from app.services.recommend import _fp, _recall, _require, card, content_profile, popular_quality
from app.services.serving import ServingState
from app.services.user_data import load_user_state

ALGORITHM = "plan_ga"
ALGORITHM_VERSION = "1.0"


def _relevance(state: ServingState, us, pool: np.ndarray) -> tuple[dict[int, float], str]:
    snap = state.snap
    elig = np.zeros(snap.n, dtype=bool)
    elig[pool] = True
    rated_cf = [i for i in us.ratings if state.cat_to_cf is not None and state.cat_to_cf[i] >= 0]
    rel: dict[int, float] = {}
    source = "popular_quality"
    if rated_cf and state.icf is not None:
        raw = _recall(state, "icf", us, rated_cf, elig)
        if raw is not None and len(raw[0]):
            for i, p in zip(raw[0].tolist(), raw[1].tolist(), strict=True):
                rel[i] = (p - 1.0) / 9.0
            source = "icf"
    prof = content_profile(snap, us)
    if prof is not None:
        from app.recommendation.collection.manual import CollectionProfileScorer

        cs, _ = CollectionProfileScorer(alpha=0.7).score(snap, prof, pool)
        floor = min(rel.values()) if rel else 1.0
        for i, s in zip(pool.tolist(), cs.tolist(), strict=True):
            if i not in rel:
                rel[i] = s * (0.99 * floor if source == "icf" else 1.0)
        source = source if source == "icf" else "content"
    if not rel:
        pq = popular_quality(snap, pool)
        rel = dict(zip(pool.tolist(), pq.tolist(), strict=True))
    else:
        missing = [i for i in pool.tolist() if i not in rel]
        if missing:
            pq = popular_quality(snap, np.array(missing))
            floor = min(rel.values())
            for i, s in zip(missing, pq.tolist(), strict=True):
                rel[i] = 0.5 * floor * s
    return rel, source


def build_problem(state: ServingState, db: Session, user: User, req: dict) -> tuple[PlanProblem, dict]:
    snap = _require(state)
    cfg = state.settings.planner
    us = load_user_state(db, user, snap)
    completed = us.items_with_status(["completed"])
    watched_eps = {i: eps for i, (st, eps) in us.progress.items()}
    completed_keys = {str(snap.keys[i]) for i in completed}
    mode = req.get("mode", "complete")
    cs = ConstraintSet(require_planner_eligible=True, include_tags=[("genre", g) for g in req.get("include_genres", [])],
                       exclude_tags=[("genre", g) for g in req.get("exclude_genres", [])],
                       min_score=req.get("min_score"), max_episodes=req.get("max_episodes_per_title"))
    cs.exclude_items = completed | set(us.ratings) | us.items_with_status(["dropped"])
    cs.sources["dedupe.excluded_items"] = "completed / rated / dropped"
    if us.excluded_items:
        cs.user_excluded_items = set(us.excluded_items)
    if us.excluded_tags:
        cs.user_excluded_tags = list(us.excluded_tags)
    elig = state.constraints.evaluate(cs)
    pool = np.flatnonzero(elig.mask)
    must: set[int] = set()
    problems = []
    for key in req.get("must_include", []):
        i = snap.index.get(key)
        if i is None or not elig.mask[i]:
            reasons = elig.fail_reasons(i) if i is not None else ["unknown title"]
            problems.append(f"{key}: cannot be planned ({', '.join(reasons)})")
        else:
            must.add(i)
    if problems:
        raise ValidationFailed("Some requested titles cannot be planned.", code="must_include_invalid",
                               details={"problems": problems})
    rel, source = _relevance(state, us, pool) if len(pool) else ({}, "none")
    ranked = sorted(pool.tolist(), key=lambda i: (-rel.get(i, 0.0), snap.keys[i]))
    cand = ranked[:int(cfg.get("candidate_pool", 40))]
    for m in must:
        if m not in cand:
            cand.append(m)
    graph = PrequelGraph(confirmed_edges(db, ("prequel", "sequel")))
    excluded: dict[str, int] = {}
    units: list[Unit] = []
    all_items: set[int] = set()
    for i in cand:
        chain, cycle, _refs = graph.chain(str(snap.keys[i]))
        unmet = [snap.index[k] for k in chain if k in snap.index and k not in completed_keys]
        unknown_prereq = [k for k in chain if k not in snap.index]
        reason = None
        if cycle:
            reason = "prerequisite_cycle"
        elif unknown_prereq:
            reason = "prerequisite_not_in_catalogue"
        elif unmet and mode == "partial":
            reason = "needs_prequel_first"
        elif unmet and not req.get("allow_prerequisites", True):
            reason = "needs_prequel_first"
        elif any(not title_time(snap, p, watched_eps.get(p, 0)).known for p in unmet):
            reason = "prerequisite_duration_unknown"
        if reason:
            if i in must:
                raise ValidationFailed(f"{snap.titles[i]} cannot be planned: {reason.replace('_', ' ')}.",
                                       code="must_include_invalid")
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        units.append(Unit(i, unmet + [i]))
        all_items.update(unmet + [i])
    times = {i: title_time(snap, i, watched_eps.get(i, 0)) for i in all_items}
    for i in list(all_items):
        rel.setdefault(i, 0.0)
    quality_arr = minmax(np.nan_to_num(snap.score[list(all_items)])) if all_items else np.zeros(0)
    quality = dict(zip(list(all_items), quality_arr.tolist(), strict=True))
    item_list = sorted(all_items)
    sim = {}
    if len(item_list) >= 2:
        S, V = state.similarity["weighted_attributes"].pairwise(snap, np.array(item_list))
        for a in range(len(item_list)):
            for b in range(a + 1, len(item_list)):
                if V[a, b]:
                    sim[(item_list[a], item_list[b])] = float(S[a, b])
    counts = snap.train_interactions
    mx = float(counts.max()) if counts is not None and len(counts) else 0.0
    novelty = {i: 1.0 - np.log1p(counts[i]) / np.log1p(mx) for i in all_items} if mx > 0 else None
    weights = dict(cfg.get("weights", DEFAULT_WEIGHTS))
    weights["diversity"] = weights.get("diversity", 0.2) * {"low": 0.5, "normal": 1.0, "high": 1.75}[
        req.get("diversity", "normal")]
    problem = PlanProblem(candidates=[u.target for u in units], units=units, times=times, relevance=rel,
                          quality=quality, novelty=novelty, sim=sim, budget=float(req["budget_minutes"]),
                          max_titles=int(req.get("max_titles", 5)), break_minutes=int(req.get("break_minutes_per_title", 0)),
                          mode=mode, must=must, weights=weights)
    not_plannable = snap.recommendable & snap.in_catalog & ~snap.planner_eligible
    ctx = {"eligible_candidates": int(len(pool)), "candidate_pool": len(units), "relevance_source": source,
           "excluded_from_pool": excluded, "rule_failures": elig.fail_counts,
           "catalogue_not_plannable": {
               "unknown_episodes": int((not_plannable & ~np.isfinite(snap.episodes)).sum()),
               "unknown_duration": int((not_plannable & ~np.isfinite(snap.duration)).sum()),
               "not_finished_airing": int((not_plannable & (snap.status != "Finished Airing")).sum())},
           "completed_keys": completed_keys, "us": us}
    return problem, ctx


def _order(problem: PlanProblem, items: list[int], requires: dict[int, list[int]]) -> list[int]:
    remaining = list(items)
    placed: list[int] = []
    while remaining:
        ready = [i for i in remaining if all(r in placed or r not in items for r in requires.get(i, []))]
        if not ready:          # cycle guard (excluded earlier); keep deterministic order
            ready = remaining
        nxt = min(ready, key=lambda i: (-problem.relevance.get(i, 0.0), i))
        placed.append(nxt)
        remaining.remove(nxt)
    return placed


def _payload(state: ServingState, problem: PlanProblem, items: list[int], episodes: dict[int, int] | None,
             req: dict, completed_keys: set[str], extra: dict) -> dict:
    snap = state.snap
    requires = {u.target: u.items[:-1] for u in problem.units}
    order = _order(problem, items, requires)
    entries: list[PlanEntry] = []
    for i in order:
        t: TitleTime = problem.times[i]
        if t.is_movie:
            s, e = 1, 1
        else:
            n = t.remaining if episodes is None else episodes[i]
            s, e = t.watched + 1, t.watched + n
        minutes = t.full_cost() if (episodes is None or t.is_movie) else t.partial_cost(episodes[i])
        entries.append(PlanEntry(i, str(snap.keys[i]), s, e, round(float(minutes), 3),
                                 is_full=(t.is_movie or e == t.total_episodes),
                                 requires=[str(snap.keys[r]) for r in requires.get(i, [])]))
    by_key = {en.item_key: en for en in entries}
    for en in entries:
        for r in en.requires:
            if r in by_key:
                by_key[r].prerequisite_of.append(en.item_key)
    times_by_key = {str(snap.keys[i]): t for i, t in problem.times.items()}
    feas = check(entries, times_by_key, problem.budget, problem.max_titles, problem.break_minutes, completed_keys,
                 problem.mode)
    titles = {str(snap.keys[i]): snap.titles[i] for i in problem.times}
    schedule = build_schedule(entries, times_by_key, titles, req.get("daily_slots", []),
                              problem.break_minutes) if req.get("daily_slots") else None
    out_items = []
    for en in entries:
        i = en.idx
        reasons = [X.reason("plan_budget", minutes=round(en.minutes), budget=int(problem.budget))]
        for r in en.requires:
            reasons.append(X.reason("plan_prereq", title=titles.get(r, r)))
        if en.prerequisite_of:
            reasons.append({"code": "plan_is_prerequisite", "params": {}, "text": "Watch this first: it comes "
                            "before " + ", ".join(titles.get(k, k) for k in en.prerequisite_of)})
        out_items.append({**card(snap, i), "episode_start": en.episode_start, "episode_end": en.episode_end,
                          "episodes_in_plan": en.episode_end - en.episode_start + 1, "minutes": round(en.minutes, 1),
                          "is_full": en.is_full, "requires": en.requires,
                          "requires_titles": [titles.get(k, k) for k in en.requires],
                          "prerequisite_of": en.prerequisite_of, "reasons": reasons,
                          "relevance": round(float(problem.relevance.get(i, 0.0)), 4),
                          "duration_evidence": problem.times[i].evidence()})
    warnings = []
    if not entries:
        warnings.append("No title fits these constraints and the time budget.")
    if schedule and not schedule["feasible"]:
        warnings.extend(schedule["problems"])
    return {"title": req.get("title") or f"{round(problem.budget / 60, 1)} h plan", "mode": problem.mode,
            "items": out_items, "total_minutes": round(feas.total_minutes, 1), "budget_minutes": problem.budget,
            "remaining_minutes": round(feas.remaining_minutes, 1), "titles": len(entries),
            "max_titles": problem.max_titles, "feasibility": feas.public(), "schedule": schedule,
            "warnings": warnings, "request": {k: v for k, v in req.items() if k != "title"},
            "dataset_mode": state.dataset_mode, "dataset_version": state.dataset_version, **extra}


def compute(state: ServingState, db: Session, user: User, req: dict) -> dict:
    t0 = time.perf_counter()
    cfg = state.settings.planner
    problem, ctx = build_problem(state, db, user, req)
    seed = int(_fp([user.id, ctx["us"].profile_version, req, state.dataset_version]), 16) % (2 ** 32)
    ga = PlanGA(population=cfg.get("population", 60), generations=cfg.get("generations", 120),
                time_budget_ms=cfg.get("time_budget_ms", 1500))
    sol = ga.solve(problem, np.random.default_rng(seed))
    baseline = greedy_complete(problem) if problem.mode == "complete" else None
    oracle = None
    if problem.mode == "complete" and 0 < len(problem.units) <= int(cfg.get("exact_oracle_max_items", 14)):
        oracle = exhaustive_complete(problem)
    objective = {"optimizer": sol.optimizer, "fitness": round(sol.fitness, 5), "terms": _round(sol.terms),
                 "weights": problem.weights, "generations": max(0, len(sol.log) - 1), "stopped": sol.stopped,
                 "greedy_baseline": None if baseline is None else {"fitness": round(baseline.fitness, 5),
                                                                   "terms": _round(baseline.terms)},
                 "exact_oracle": None if oracle is None else {"fitness": round(oracle.fitness, 5),
                                                              "gap": round(oracle.fitness - sol.fitness, 6)}}
    diag = {k: v for k, v in ctx.items() if k not in ("us", "completed_keys")}
    payload = _payload(state, problem, sol.items, sol.episodes, req, ctx["completed_keys"],
                       {"objective": objective, "diagnostics": diag, "algorithm": ALGORITHM,
                        "algorithm_version": ALGORITHM_VERSION,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    if sol.stopped == "infeasible":
        payload["warnings"].append("The required titles alone do not fit the budget / title limit.")
    if payload["items"] and not payload["feasibility"]["feasible"]:
        # the optimiser must never return an unchecked plan: fall back to the validated greedy plan
        from app.core.logging import get_logger, log_event

        log_event(get_logger("planner"), "optimiser produced an infeasible plan",
                  problems=payload["feasibility"]["problems"][:3])
        fallback_items = baseline.items if baseline is not None else []
        payload = _payload(state, problem, fallback_items, None, req, ctx["completed_keys"],
                           {"objective": {**objective, "fallback": "greedy_density"}, "diagnostics": diag,
                            "algorithm": "greedy_density", "algorithm_version": ALGORITHM_VERSION,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return payload


def _round(d: dict) -> dict:
    return {k: (None if v is None else round(float(v), 5)) for k, v in (d or {}).items()}


# ---------------------------------------------------------------------------- persistence


def _owned(db: Session, user: User, plan_id: int) -> ViewingPlan:
    p = db.get(ViewingPlan, plan_id)
    if p is None or p.user_id != user.id:
        raise NotFound("Plan not found.", code="plan_not_found")
    return p


def _store_revision(db: Session, plan: ViewingPlan, revision: int, payload: dict, note: str) -> PlanRevision:
    rev = PlanRevision(plan_id=plan.id, revision=revision, request=payload["request"],
                       summary={k: payload[k] for k in ("title", "mode", "total_minutes", "budget_minutes",
                                                        "remaining_minutes", "titles", "max_titles")}
                       | {"objective": payload.get("objective"), "diagnostics": payload.get("diagnostics"),
                          "warnings": payload.get("warnings", [])},
                       feasibility=payload["feasibility"], algorithm=payload.get("algorithm", ALGORITHM),
                       algorithm_version=payload.get("algorithm_version", ALGORITHM_VERSION),
                       dataset_version=payload["dataset_version"] or "none", change_note=note)
    db.add(rev)
    db.flush()
    for n, it in enumerate(payload["items"]):
        db.add(PlanItem(revision_id=rev.id, order_index=n, item_key=it["item_key"], episode_start=it["episode_start"],
                        episode_end=it["episode_end"], minutes=it["minutes"], is_full=it["is_full"],
                        reasons=it["reasons"], prerequisite_keys=it["requires"],
                        duration_evidence=it["duration_evidence"]))
    if payload.get("schedule"):
        for s in payload["schedule"]["sessions"]:
            db.add(PlanSession(revision_id=rev.id, session_index=s["index"], label=s["label"],
                               capacity_minutes=s["capacity_minutes"], entries=s["entries"]))
    plan.current_revision = revision
    plan.updated_at = utcnow()
    return rev


def save(state: ServingState, db: Session, user: User, req: dict) -> dict:
    payload = compute(state, db, user, req)
    plan = ViewingPlan(user_id=user.id, title=payload["title"][:120])
    db.add(plan)
    db.flush()
    _store_revision(db, plan, 1, payload, "created")
    return {"id": plan.id, "revision": 1, **payload}


def list_plans(db: Session, user: User) -> list[dict]:
    plans = db.scalars(select(ViewingPlan).where(ViewingPlan.user_id == user.id)
                       .order_by(ViewingPlan.updated_at.desc())).all()
    out = []
    for p in plans:
        rev = db.scalar(select(PlanRevision).where(PlanRevision.plan_id == p.id, PlanRevision.revision == p.current_revision))
        out.append({"id": p.id, "title": p.title, "revision": p.current_revision, "updated_at": p.updated_at.isoformat(),
                    "summary": rev.summary if rev else {}, "feasible": (rev.feasibility or {}).get("feasible") if rev else None})
    return out


def get_plan(state: ServingState, db: Session, user: User, plan_id: int, revision: int | None = None) -> dict:
    plan = _owned(db, user, plan_id)
    rev_no = revision or plan.current_revision
    rev = db.scalar(select(PlanRevision).where(PlanRevision.plan_id == plan.id, PlanRevision.revision == rev_no))
    if rev is None:
        raise NotFound("Revision not found.", code="revision_not_found")
    snap = state.snap
    items = []
    for it in rev.items:
        i = snap.index.get(it.item_key) if snap is not None else None
        base = card(snap, i) if i is not None else {"item_key": it.item_key, "title": it.item_key}
        items.append({**base, "episode_start": it.episode_start, "episode_end": it.episode_end,
                      "episodes_in_plan": it.episode_end - it.episode_start + 1, "minutes": it.minutes,
                      "is_full": it.is_full, "requires": it.prerequisite_keys,
                      "requires_titles": [snap.titles[snap.index[k]] if snap and k in snap.index else k
                                          for k in it.prerequisite_keys],
                      "prerequisite_of": [], "reasons": it.reasons, "duration_evidence": it.duration_evidence})
    keyset = {x["item_key"] for x in items}
    for x in items:
        for r in x["requires"]:
            if r in keyset:
                next(y for y in items if y["item_key"] == r)["prerequisite_of"].append(x["item_key"])
    s = rev.summary or {}
    schedule = None
    if rev.sessions:
        schedule = {"sessions": [{"index": ss.session_index, "label": ss.label, "capacity_minutes": ss.capacity_minutes,
                                  "used_minutes": round(sum(e["minutes"] for e in ss.entries), 2),
                                  "entries": ss.entries} for ss in rev.sessions],
                    "feasible": not any(p for p in (s.get("warnings") or []) if "session" in p.lower()),
                    "problems": [p for p in (s.get("warnings") or []) if "session" in p.lower()]}
    revisions = [{"revision": r.revision, "created_at": r.created_at.isoformat(), "change_note": r.change_note,
                  "feasible": (r.feasibility or {}).get("feasible")} for r in plan.revisions]
    return {"id": plan.id, "title": plan.title, "revision": rev.revision, "current_revision": plan.current_revision,
            "revisions": revisions, "mode": s.get("mode"), "items": items, "total_minutes": s.get("total_minutes"),
            "budget_minutes": s.get("budget_minutes"), "remaining_minutes": s.get("remaining_minutes"),
            "titles": s.get("titles"), "max_titles": s.get("max_titles"), "feasibility": rev.feasibility,
            "schedule": schedule, "warnings": s.get("warnings", []), "objective": s.get("objective"),
            "diagnostics": s.get("diagnostics"), "request": rev.request, "algorithm": rev.algorithm,
            "algorithm_version": rev.algorithm_version, "dataset_version": rev.dataset_version,
            "dataset_mode": state.dataset_mode, "created_at": rev.created_at.isoformat()}


def edit(state: ServingState, db: Session, user: User, plan_id: int, title: str | None, remove: list[str],
         edits: list[dict], order: list[str] | None) -> dict:
    """Manual edits create a new revision whose feasibility is recomputed from scratch."""
    snap = _require(state)
    plan = _owned(db, user, plan_id)
    cur = get_plan(state, db, user, plan_id)
    req = dict(cur["request"])
    us = load_user_state(db, user, snap)
    watched = {str(snap.keys[i]): eps for i, (st, eps) in us.progress.items()}
    completed_keys = {str(snap.keys[i]) for i in us.items_with_status(["completed"])}
    items = [it for it in cur["items"] if it["item_key"] not in set(remove)]
    edit_map = {e["item_key"]: e for e in edits}
    if order:
        pos = {k: n for n, k in enumerate(order)}
        items.sort(key=lambda it: pos.get(it["item_key"], len(pos)))
    times = {}
    entries = []
    for it in items:
        i = snap.index.get(it["item_key"])
        if i is None:
            raise ValidationFailed(f"{it['item_key']} is no longer in the catalogue.", code="plan_item_missing")
        t = title_time(snap, i, watched.get(it["item_key"], 0))
        times[it["item_key"]] = t
        s, e = it["episode_start"], it["episode_end"]
        if it["item_key"] in edit_map:
            ed = edit_map[it["item_key"]]
            s = ed.get("episode_start") or s
            e = ed.get("episode_end") or e
        if t.is_movie:
            minutes = t.runtime or 0.0
        else:
            minutes = max(0, e - s + 1) * (t.per_episode or 0.0)
        entries.append(PlanEntry(i, it["item_key"], s, e, round(float(minutes), 3),
                                 is_full=(t.is_movie or e == t.total_episodes), requires=list(it.get("requires", []))))
    mode = req.get("mode", "complete")
    if edit_map and mode == "complete" and any(not en.is_full for en in entries):
        mode = "partial"
        req["mode"] = "partial"
    feas = check(entries, times, float(req["budget_minutes"]), int(req.get("max_titles", 5)),
                 int(req.get("break_minutes_per_title", 0)), completed_keys, mode)
    titles = {k: snap.titles[snap.index[k]] for k in times}
    schedule = build_schedule(entries, times, titles, req.get("daily_slots", []),
                              int(req.get("break_minutes_per_title", 0))) if req.get("daily_slots") else None
    warnings = list(feas.problems) + ([] if not schedule or schedule["feasible"] else schedule["problems"])
    payload = {"title": title or plan.title, "mode": mode, "request": req, "total_minutes": round(feas.total_minutes, 1),
               "budget_minutes": float(req["budget_minutes"]), "remaining_minutes": round(feas.remaining_minutes, 1),
               "titles": len(entries), "max_titles": int(req.get("max_titles", 5)), "feasibility": feas.public(),
               "schedule": schedule, "warnings": warnings, "dataset_version": state.dataset_version,
               "algorithm": "manual_edit", "algorithm_version": ALGORITHM_VERSION,
               "objective": None, "diagnostics": None,
               "items": [{"item_key": en.item_key, "episode_start": en.episode_start, "episode_end": en.episode_end,
                          "minutes": round(en.minutes, 1), "is_full": en.is_full, "requires": en.requires,
                          "reasons": [X.reason("plan_budget", minutes=round(en.minutes), budget=int(req["budget_minutes"]))],
                          "duration_evidence": times[en.item_key].evidence()} for en in entries]}
    if title:
        plan.title = title[:120]
    note = "edited: " + ", ".join(filter(None, [f"removed {len(remove)}" if remove else "",
                                                 f"changed {len(edit_map)}" if edit_map else "",
                                                 "reordered" if order else "", "renamed" if title else ""]))
    _store_revision(db, plan, plan.current_revision + 1, payload, note[:200])
    db.flush()
    return get_plan(state, db, user, plan_id)


def reoptimize(state: ServingState, db: Session, user: User, plan_id: int, new_request: dict | None) -> dict:
    plan = _owned(db, user, plan_id)
    cur = get_plan(state, db, user, plan_id)
    req = {**cur["request"], **(new_request or {})}
    payload = compute(state, db, user, req)
    _store_revision(db, plan, plan.current_revision + 1, payload, "re-optimised")
    db.flush()
    return get_plan(state, db, user, plan_id)


def delete(db: Session, user: User, plan_id: int) -> None:
    db.delete(_owned(db, user, plan_id))


def export(state: ServingState, db: Session, user: User, plan_id: int, fmt: str) -> tuple[str, str, str]:
    plan = get_plan(state, db, user, plan_id)
    safe = {k: v for k, v in plan.items() if k not in ("diagnostics",)}
    stamp = datetime.utcnow().strftime("%Y%m%d")
    if fmt == "json":
        return plan_export.to_json(safe), "application/json", f"anilogic-plan-{plan_id}-{stamp}.json"
    if fmt == "markdown":
        return plan_export.to_markdown(safe), "text/markdown; charset=utf-8", f"anilogic-plan-{plan_id}-{stamp}.md"
    if fmt == "html":
        return plan_export.to_html(safe), "text/html; charset=utf-8", f"anilogic-plan-{plan_id}-{stamp}.html"
    raise ValidationFailed("format must be json, markdown or html", code="invalid_format")
