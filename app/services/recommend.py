"""Recommendation surfaces: similar titles (1), collection-based (2), personalised home (3).

All surfaces share the ConstraintEngine, return actual_k <= requested_k with the
reason when fewer titles qualify (never padded with duplicates or ineligible
titles), and record a recommendation run (versions, candidates, results).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.core.errors import CapabilityUnavailable, NotFound, ValidationFailed
from app.db.models import Anime, RecommendationRun, User
from app.recommendation import explanations as X
from app.recommendation.constraints import ConstraintSet
from app.recommendation.features import CatalogSnapshot, minmax, stable_order
from app.recommendation.profiles import CollectionProfile, UserState, build_collection_profile
from app.recommendation.rerank.ga import GeneticListOptimizer, relevance_greedy
from app.recommendation.rerank.objective import ListObjective, fresh_flags, novelty_scores
from app.services.serving import ServingState
from app.services.user_data import load_user_state

# ------------------------------------------------------------------------ cache


class LRU:
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return copy.deepcopy(self._d[key])
        return None

    def put(self, key, value) -> None:
        self._d[key] = copy.deepcopy(value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()


CACHE = LRU()
_FOLD_IN = LRU(2048)


def _fp(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# ------------------------------------------------------------------------ helpers


def card(snap: CatalogSnapshot, i: int) -> dict:
    ep = snap.episodes[i]
    sc = snap.score[i]
    return {"item_key": str(snap.keys[i]), "title": snap.titles[i],
            "title_english": snap.titles_english[i] if snap.titles_english else None,
            "cover_url": snap.covers[i] if snap.covers else None, "type": snap.type[i],
            "episodes": None if not np.isfinite(ep) else int(ep),
            "catalog_score": None if not np.isfinite(sc) else round(float(sc), 2),
            "genres": snap.tags["genre"].tags_of(i) if "genre" in snap.tags else [],
            "year": None if not np.isfinite(snap.year[i]) else int(snap.year[i])}


def _meta(state: ServingState, requested_k: int, actual_k: int, algorithm: str, model_version: str | None,
          fallback: str | None, warnings: list[str]) -> dict:
    return {"requested_k": requested_k, "actual_k": actual_k, "dataset_mode": state.dataset_mode,
            "dataset_version": state.dataset_version, "algorithm": algorithm, "model_version": model_version,
            "fallback": fallback, "warnings": warnings}


def log_run(db: Session, request_id: str, user: User | None, surface: str, params: dict, algorithm: str,
            model_version: str | None, state: ServingState, requested_k: int, items: list[dict],
            candidates: list[dict], trace: dict) -> None:
    db.add(RecommendationRun(request_id=request_id, user_id=user.id if user else None, surface=surface,
                             params_fingerprint=_fp(params), algorithm=algorithm, model_version=model_version,
                             dataset_version=state.dataset_version or "none", requested_k=requested_k,
                             actual_k=len(items), candidates=candidates[:50],
                             results=[{"item_key": it["item_key"], "score": it.get("score")} for it in items],
                             trace=trace))


def _require(state: ServingState) -> CatalogSnapshot:
    if state.snap is None:
        cap = state.cap("catalog")
        raise CapabilityUnavailable(cap.detail, code=cap.status)
    return state.snap


def served_index(state: ServingState, item_key: str) -> int:
    snap = _require(state)
    i = snap.index.get(item_key)
    if i is None or not snap.in_catalog[i] or snap.explicit[i]:
        raise NotFound("Anime not found.", code="anime_not_found")
    return i


def _account_rules(cs: ConstraintSet, us: UserState) -> ConstraintSet:
    if us.excluded_items:
        cs.user_excluded_items = set(us.excluded_items)
        cs.sources["user.excluded_item"] = "account"
    if us.excluded_tags:
        cs.user_excluded_tags = list(us.excluded_tags)
        cs.sources["user.excluded_tag"] = "account"
    return cs


# ------------------------------------------------------------------------ (1) similar


def similar(state: ServingState, db: Session, user: User | None, item_key: str, k: int | None = None,
            scorer_id: str | None = None, hide_watched: bool | None = None) -> dict:
    t0 = time.perf_counter()
    snap = _require(state)
    cfg = state.settings.recommendation.get("similar", {})
    k = int(k or cfg.get("k", 5))
    if not 1 <= k <= int(cfg.get("max_k", 10)):
        raise ValidationFailed(f"k must be between 1 and {cfg.get('max_k', 10)}.", code="invalid_k")
    target = served_index(state, item_key)
    requested = scorer_id or cfg.get("scorer", "weighted_attributes")
    warnings, fallback = [], None
    scorer = state.similarity.get(requested)
    if scorer is None:
        fallback = f"{requested} unavailable -> weighted_attributes"
        scorer = state.similarity["weighted_attributes"]
    us = load_user_state(db, user, snap)
    if hide_watched is None:
        from app.db.models import Preference

        pref = db.get(Preference, user.id) if user else None
        hide_watched = bool(pref.hide_watched_in_similar) if pref else bool(cfg.get("hide_watched_default", False))
    key = ("similar", state.dataset_version, item_key, scorer.scorer_id, k, bool(hide_watched),
           user.id if user else None, us.profile_version, state.settings.fingerprint("recommendation", "similar"))
    cached = CACHE.get(key)
    if cached is not None:
        cached["request_id"] = new_request_id()
        cached["meta"]["cached"] = True
        return cached
    cs = ConstraintSet(exclude_items={target})
    cs.sources["dedupe.excluded_items"] = "surface: the target title itself"
    _account_rules(cs, us)
    if hide_watched and us.user_id:
        cs.exclude_items |= set(us.ratings) | us.items_with_status(["completed"])
        cs.sources["dedupe.excluded_items"] += " + watched/rated (your setting)"
    elig = state.constraints.evaluate(cs)
    cand = np.flatnonzero(elig.mask)
    res = scorer.score(snap, target, cand)
    valid = np.isfinite(res.score)
    min_cov = cfg.get("min_coverage_for_ranking")
    if min_cov is not None:
        valid &= res.coverage >= float(min_cov)
    insufficient = int((~np.isfinite(res.score)).sum())
    pos = np.flatnonzero(valid)
    order = pos[stable_order(res.score[pos], snap.keys[cand[pos]], res.coverage[pos])][:k]
    weights = getattr(scorer, "weights", None) or getattr(getattr(scorer, "base", None), "weights", {})
    items = []
    low_cov = float(cfg.get("low_coverage_warning", 0.5))
    for p in order:
        i = int(cand[p])
        reasons = X.similarity_reasons(snap, target, i, res.available, res.sims, int(p))
        if res.coverage[p] < low_cov:
            reasons.append(X.reason("low_coverage"))
        comps = {"coverage": round(float(res.coverage[p]), 3)}
        if weights:
            comps["dimensions"] = {d: (None if not res.available[d][p] else round(float(res.sims[d][p]), 4))
                                   for d in ("genre", "theme", "demographic", "studio", "length")}
        items.append({**card(snap, i), "score": round(float(res.score[p]), 4), "reasons": reasons,
                      "evidence_refs": [], "components": comps, "candidate_source": scorer.scorer_id,
                      "constraints_passed": True})
    if len(items) < k:
        warnings.append(f"Only {len(items)} titles have comparable metadata and pass your filters.")
    target_missing = [X.DIM_LABEL[d] for d in ("theme", "demographic", "studio", "length")
                      if not X.target_has(snap, target, d)]
    if target_missing:
        warnings.append(f"This title has no {', '.join(target_missing)} data, so those attributes could not be "
                        "compared.")
    if res.coverage[order].size and float(res.coverage[order].mean()) < low_cov:
        warnings.append("Metadata for this title is sparse; similarity is based on few attributes.")
    request_id = new_request_id()
    out = {"request_id": request_id, "target": card(snap, target), "items": items,
           "meta": _meta(state, k, len(items), scorer.scorer_id, getattr(scorer, "version", None), fallback, warnings)
           | {"insufficient_metadata_candidates": insufficient, "scorer_options": sorted(state.similarity),
              "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    log_run(db, request_id, user, "similar", {"target": item_key, "k": k, "scorer": scorer.scorer_id}, scorer.scorer_id,
            getattr(scorer, "version", None), state, k, items, [], {"eligible": elig.count, "hide_watched": hide_watched})
    CACHE.put(key, out)
    return out


# ------------------------------------------------------------------------ (2) collection


@dataclass
class CollectionFilters:
    keywords: list[str]
    include_tags: list[tuple[str, str]]
    min_score: float | None
    max_episodes: int | None
    types: list[str] | None

    def public(self) -> dict:
        return {"keywords": self.keywords, "include_tags": [list(t) for t in self.include_tags],
                "min_score": self.min_score, "max_episodes": self.max_episodes, "types": self.types}


def collection(state: ServingState, db: Session, user: User, collection_id: int, filters: CollectionFilters,
               k: int | None = None, scorer_id: str | None = None, relaxation: dict | None = None) -> dict:
    from app.services.user_data import get_collection

    t0 = time.perf_counter()
    snap = _require(state)
    cfg = state.settings.recommendation.get("collection", {})
    k = int(k or cfg.get("k", 8))
    if not 1 <= k <= 20:
        raise ValidationFailed("k must be between 1 and 20.", code="invalid_k")
    coll, rows = get_collection(db, user, collection_id)
    items = [snap.index[a.item_key] for _, a in rows if a.item_key in snap.index]
    us = load_user_state(db, user, snap)
    requested = scorer_id or cfg.get("scorer", "collection_profile_v1")
    fallback = None
    scorer = state.collection_scorers.get(requested)
    if requested == "two_tower_v1" and state.cap("two_tower").status != "ready":
        scorer, fallback = None, "two_tower_v1 is not deployed (limited_by_data)"
    if scorer is None:
        fallback = fallback or f"{requested} unavailable"
        fallback += " -> collection_profile_v1"
        scorer = state.collection_scorers["collection_profile_v1"]
    profile = build_collection_profile(snap, items, us.ratings, us.rating_mean, us.excluded_items)
    request_id = new_request_id()
    base_meta = {"collection": {"id": coll.id, "name": coll.name, "size": len(items)},
                 "profile": _profile_summary(snap, profile), "filters": filters.public(),
                 "relaxation_applied": relaxation}
    if profile.empty:
        out = {"request_id": request_id, "items": [], "status": "empty_profile", "relaxation_options": [],
               "constraints_satisfied": True, **base_meta,
               "meta": _meta(state, k, 0, scorer.scorer_id, scorer.version, fallback,
                             ["This collection has no positive titles yet. Add a few titles you like to get "
                              "collection-based suggestions."])}
        return out
    cs = ConstraintSet(min_score=filters.min_score, max_episodes=filters.max_episodes, types=filters.types,
                       include_tags=filters.include_tags, keywords=filters.keywords)
    for rid, v in (("filter.min_score", filters.min_score), ("filter.max_episodes", filters.max_episodes),
                   ("filter.types", filters.types), ("filter.include_tags", filters.include_tags),
                   ("filter.keywords", filters.keywords)):
        if v not in (None, []):
            cs.sources[rid] = "relaxed by you" if relaxation and relaxation.get("rule_id") == rid else "your filter"
    cs.exclude_items = set(items) | us.items_with_status(list(cfg.get("exclude_statuses", ["completed"])))
    cs.sources["dedupe.excluded_items"] = "already in this collection / completed"
    _account_rules(cs, us)
    elig = state.constraints.evaluate(cs)
    warnings = []
    if elig.count == 0:
        options = state.constraints.suggest_relaxations(cs, k)
        warnings.append("No titles meet all of the selected filters." if cs.active_rules() else
                        "No eligible titles.")
        if elig.unknown_tag_namespaces:
            warnings.append("Titles with unknown tags cannot be verified against tag filters and were excluded.")
        return {"request_id": request_id, "items": [], "status": "no_match", "constraints_satisfied": True,
                "relaxation_options": options, "rule_failures": elig.fail_counts, **base_meta,
                "meta": _meta(state, k, 0, scorer.scorer_id, scorer.version, fallback, warnings)}
    cand = np.flatnonzero(elig.mask)
    scores, comps = scorer.score(snap, profile, cand)
    order = stable_order(scores, snap.keys[cand])[:k]
    out_items = []
    met = [f for f, v in (("keywords " + "/".join(filters.keywords), filters.keywords),
                          (f"score >= {filters.min_score}", filters.min_score),
                          (f"<= {filters.max_episodes} episodes", filters.max_episodes),
                          ("tags " + "/".join(t[1] for t in filters.include_tags), filters.include_tags),
                          ("type " + "/".join(filters.types or []), filters.types)) if v not in (None, [])]
    for p in order:
        i = int(cand[p])
        reasons = X.profile_reasons(snap, profile, i)
        if met:
            reasons.append(X.reason("filters_met", filters=", ".join(met)))
        c = {name: (None if not np.isfinite(v[p]) else round(float(v[p]), 4)) for name, v in comps.items()}
        out_items.append({**card(snap, i), "score": round(float(scores[p]), 4), "reasons": reasons,
                          "evidence_refs": [], "components": c, "candidate_source": scorer.scorer_id,
                          "constraints_passed": True})
    if len(out_items) < k:
        warnings.append(f"Only {len(out_items)} titles meet the filters.")
    out = {"request_id": request_id, "items": out_items, "status": "ok", "constraints_satisfied": True,
           "relaxation_options": [], "rule_failures": elig.fail_counts, **base_meta,
           "meta": _meta(state, k, len(out_items), scorer.scorer_id, scorer.version, fallback, warnings)
           | {"eligible_candidates": elig.count, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    log_run(db, request_id, user, "collection", {"collection": coll.id, **filters.public(), "k": k},
            scorer.scorer_id, scorer.version, state, k, out_items, [], {"eligible": elig.count,
                                                                         "rule_failures": elig.fail_counts})
    return out


def _profile_summary(snap: CatalogSnapshot, profile: CollectionProfile) -> dict:
    out = {ns: profile.top_tags(snap, ns, 5) for ns in ("genre", "theme", "demographic", "studio")}
    out["typical_episodes"] = None if profile.log_eps_mean is None else round(float(np.expm1(profile.log_eps_mean)), 1)
    out["seed_items"] = len(profile.items)
    out["skipped_items"] = [{"item_key": str(snap.keys[i]), "reason": r} for i, r in profile.skipped]
    return out


# ------------------------------------------------------------------------ (3) home


def content_profile(snap: CatalogSnapshot, us: UserState) -> CollectionProfile | None:
    pos = us.positive_items()
    prof = build_collection_profile(snap, pos) if pos else None
    if us.liked_tags:
        if prof is None:
            prof = build_collection_profile(snap, [])
        for ns, value in us.liked_tags:
            tm = snap.tags.get(ns)
            if tm is None:
                continue
            from app.core.text import norm

            col = tm.col.get(norm(value))
            if col is None:
                continue
            f = prof.tag_freq.setdefault(ns, np.zeros(len(tm.vocab)))
            f[col] = max(f[col], 1.0)
            prof.ns_support[ns] = max(prof.ns_support.get(ns, 0), 1)
        prof.items = prof.items or []
    if prof is None or (not prof.items and not us.liked_tags):
        return None
    return prof


def popular_quality(snap: CatalogSnapshot, cand: np.ndarray) -> np.ndarray:
    pop = minmax(np.log1p(np.nan_to_num(snap.members[cand])))
    q = minmax(np.nan_to_num(snap.score[cand]))
    return 0.5 * pop + 0.5 * q


def home_content_scores(snap: CatalogSnapshot, prof, pool: np.ndarray, popularity_weight: float) -> np.ndarray:
    """Content-profile score for the home feed's cold / partial modes, shared with the evaluation.

    (1 - g) * profile match + g * popular_quality. Pure tag matching tends to surface obscure titles; g > 0
    keeps the list quality-controlled (the brief's cold-start rule) and is tuned on validation.
    """
    from app.recommendation.collection.manual import CollectionProfileScorer

    scores, _ = CollectionProfileScorer(alpha=0.7).score(snap, prof, pool)
    g = float(popularity_weight)
    if g > 0 and len(pool):
        scores = (1.0 - g) * np.asarray(scores) + g * popular_quality(snap, pool)
    return np.asarray(scores)


def home(state: ServingState, db: Session, user: User, k: int | None = None, engine: str | None = None,
         reranker: str | None = None, candidate_k: int | None = None) -> dict:
    t0 = time.perf_counter()
    snap = _require(state)
    cfg = state.settings.recommendation.get("home", {})
    k = int(k or cfg.get("k", 10))
    candidate_k = int(candidate_k or cfg.get("candidate_k", 20))
    if not 1 <= k <= 30 or not k <= candidate_k <= 200:
        raise ValidationFailed("Invalid k / candidate_k.", code="invalid_k")
    engine = engine or cfg.get("recall_engine", "icf")
    reranker = reranker or cfg.get("reranker", "ga")
    if engine not in ("icf", "mf", "ncf") or reranker not in ("ga", "greedy", "none"):
        raise ValidationFailed("Unknown engine / reranker.", code="invalid_algorithm")
    us = load_user_state(db, user, snap)
    key = ("home", user.id, us.profile_version, state.dataset_version, engine, reranker, k, candidate_k,
           state.settings.fingerprint("recommendation"))
    cached = CACHE.get(key)
    if cached is not None:
        cached["request_id"] = new_request_id()
        cached["meta"]["cached"] = True
        return cached
    warnings: list[str] = []
    fallback = None
    cs = ConstraintSet(exclude_items=set(us.ratings) | us.items_with_status(list(cfg.get("exclude_statuses",
                                                                                          ["completed"]))))
    cs.sources["dedupe.excluded_items"] = "already rated / completed"
    _account_rules(cs, us)
    elig = state.constraints.evaluate(cs)
    elig_mask = elig.mask
    gate = int(cfg.get("warm_gate", 8))
    rated_cf = [i for i in us.ratings if state.cat_to_cf is not None and state.cat_to_cf[i] >= 0]
    n_usable = len(rated_cf)
    mode = "warm" if n_usable >= gate else ("partial" if n_usable > 0 else "cold")
    cands: dict[int, dict] = {}
    model_version = state.dataset_version
    used_engine = engine

    if engine == "ncf" and n_usable > 0:
        fallback = "ncf: unsupported_new_user (ID-only model) -> icf"
        used_engine = "icf"
    if mode != "cold":
        raw = _recall(state, used_engine, us, rated_cf, elig_mask)
        if raw is None and used_engine != "icf":
            fallback = (fallback or "") + f"{used_engine}: insufficient support -> icf"
            used_engine = "icf"
            raw = _recall(state, "icf", us, rated_cf, elig_mask)
        if raw is not None:
            ids, pred, support = raw
            if len(ids) == 0 and mode == "warm":
                warnings.append("Your rated titles have no neighbours in the model yet; using your saved / liked "
                                "tags instead.")
                mode = "partial"
            for i, p, s in zip(ids[:candidate_k * 3].tolist(), pred[:candidate_k * 3].tolist(),
                               support[:candidate_k * 3].tolist(), strict=True):
                cands[i] = {"source": used_engine, "raw": p, "support": s}
    # content / exploration candidates (cold start, partial mix, or filling a short CF pool)
    prof = content_profile(snap, us)
    pool = np.flatnonzero(elig_mask)
    content_scores = None
    content_pop_w = float(cfg.get("content_popularity_weight", 0.0))
    if prof is not None and len(pool):
        content_scores = home_content_scores(snap, prof, pool, content_pop_w)
    rel = {}
    cf_ids = sorted(cands, key=lambda i: (-cands[i]["raw"], -cands[i]["support"], snap.keys[i]))[:candidate_k]
    if mode == "warm":
        for i in cf_ids:
            rel[i] = (cands[i]["raw"] - 1.0) / 9.0
    elif mode == "partial":
        lam = n_usable / gate
        cf_norm = {i: (cands[i]["raw"] - 1.0) / 9.0 for i in cf_ids}
        cont = {} if content_scores is None else dict(zip(pool.tolist(), content_scores.tolist(), strict=True))
        union = set(cf_ids) | set(sorted(cont, key=lambda i: (-cont[i], snap.keys[i]))[:candidate_k])
        for i in union:
            rel[i] = lam * cf_norm.get(i, 0.0) + (1 - lam) * cont.get(i, 0.0)
        for i in union - set(cf_ids):
            cands.setdefault(i, {"source": "content", "raw": cont.get(i, 0.0), "support": 0})
        for i in set(cf_ids):
            cands[i]["source"] = f"{used_engine}+content"
    else:
        if content_scores is not None:
            for i, s in zip(pool.tolist(), content_scores.tolist(), strict=True):
                rel[i] = s
                cands[i] = {"source": "content", "raw": s, "support": 0}
        else:
            if len(pool):
                pq = popular_quality(snap, pool)
                q_ok = np.nan_to_num(snap.score[pool]) >= np.nanmedian(snap.score[snap.recommendable])
                for i, s, ok in zip(pool.tolist(), pq.tolist(), q_ok.tolist(), strict=True):
                    if ok:
                        rel[i] = s
                        cands[i] = {"source": "popular_quality", "raw": s, "support": 0}
            warnings.append("No ratings or preferences yet: showing a quality-controlled popular exploration list.")
    # keep the top candidate_k by relevance; fill from content / popularity if short
    chosen = sorted(rel, key=lambda i: (-rel[i], snap.keys[i]))[:candidate_k]
    if len(chosen) < candidate_k and len(pool):
        fill_pool = [i for i in pool.tolist() if i not in set(chosen)]
        if fill_pool:
            base = content_scores if content_scores is not None else None
            if base is not None:
                cmap = dict(zip(pool.tolist(), content_scores.tolist(), strict=True))
                fs = np.array([cmap[i] for i in fill_pool])
                src = "content_fill"
            else:
                fs = popular_quality(snap, np.array(fill_pool))
                src = "popular_fill"
            floor = min((rel[i] for i in chosen), default=1.0)
            order = np.lexsort((snap.keys[fill_pool], -fs))[:candidate_k - len(chosen)]
            for o in order:
                i = fill_pool[int(o)]
                rel[i] = 0.99 * floor * float(fs[o]) if chosen else float(fs[o])
                cands[i] = {"source": src, "raw": float(fs[o]), "support": 0}
                chosen.append(i)
            if mode == "warm" and len(cf_ids) < candidate_k:
                warnings.append(f"The {used_engine.upper()} model supplied {len(cf_ids)} candidates; the rest are "
                                f"{src.replace('_', ' ')} candidates ranked below them.")
    chosen = np.array(chosen, dtype=np.int64)
    if len(chosen) == 0:
        return {"request_id": new_request_id(), "items": [], "mode": mode, "exploration": [],
                "meta": _meta(state, k, 0, "none", None, fallback, warnings + ["No eligible titles remain."])}
    relevance = np.clip(np.array([rel[i] for i in chosen]), 0, 1)
    sim, valid = state.similarity["weighted_attributes"].pairwise(snap, chosen)
    hist = us.history_tags(snap)
    obj = ListObjective(relevance, sim, valid, novelty_scores(snap.train_interactions, chosen),
                        fresh_flags(snap, chosen, hist), dict(state.settings.recommendation.get("ga", {}).get(
                            "weights", {"relevance": 0.4, "diversity": 0.3, "novelty": 0.2, "fresh": 0.1})))
    tiebreak = np.argsort(np.argsort(snap.keys[chosen]))
    greedy = relevance_greedy(obj, k, tiebreak)
    if reranker == "ga":
        seed = int(_fp([user.id, us.profile_version, state.dataset_version]), 16) % (2 ** 32)
        ga_cfg = state.settings.recommendation.get("ga", {})
        result = GeneticListOptimizer(**ga_cfg).optimize(obj, k, np.random.default_rng(seed), tiebreak=tiebreak)
    elif reranker == "greedy":
        result = greedy
    else:
        sel = np.arange(min(k, len(chosen)))   # recall order truncated
        from app.recommendation.rerank.ga import ListResult

        result = ListResult("no_rerank", sel, float(obj.fitness(sel[None, :])[0]), obj.summary(sel))
    sel = result.selected
    base_fit = float(obj.fitness(sel[None, :])[0])
    items = []
    user_mean = us.rating_mean or 0.0
    if state.icf is not None and us.rating_mean is not None:     # the same reference level as the prediction
        user_mean = state.icf.reference_mean(us.rating_mean, len(us.ratings))
    top_rel_unused = [p for p in np.argsort(-relevance, kind="stable") if p not in set(sel.tolist())]
    for pos in sel.tolist():
        i = int(chosen[pos])
        c = cands[i]
        reasons = []
        if c["source"].startswith("icf") and state.icf is not None:
            exp = state.icf.explain(int(state.cat_to_cf[i]), np.array([state.cat_to_cf[j] for j in rated_cf]),
                                    np.array([us.ratings[j] for j in rated_cf]), user_mean, top=1)
            for e in exp:
                if e["contribution"] > 0:
                    j = int(state.cf_items[e["item"]])
                    reasons.append(X.reason("icf_support", title=snap.titles[j], rating=int(us.ratings[j]),
                                            sim=round(float(e["similarity"]), 2), co=e["co_rated_users"]))
        elif c["source"].startswith("mf"):
            reasons.append(X.reason("mf_support"))
        if c["source"] in ("content", "content_fill") or c["source"].endswith("+content"):
            if prof is not None:
                pr = X.profile_reasons(snap, prof, i)
                if pr:
                    tags = pr[0]["params"].get("tags", "")
                    reasons.append(X.reason("content_profile", tags=tags))
            if content_pop_w > 0 and not c["source"].startswith(("icf", "mf")):
                reasons.append(X.reason("content_popular", weight=round(content_pop_w, 2)))
        if c["source"] in ("popular_quality", "popular_fill"):
            reasons.append(X.reason("popular_quality"))
        comps = {"relevance": round(float(relevance[pos]), 4), "candidate_raw": round(float(c["raw"]), 4)}
        if obj.novelty is not None:
            comps["novelty"] = round(float(obj.novelty[pos]), 4)
            if obj.novelty[pos] >= 0.6:
                reasons.append(X.reason("novel", nov=round(float(obj.novelty[pos]), 2)))
        if obj.fresh is not None:
            comps["fresh"] = bool(obj.fresh[pos])
            if obj.fresh[pos]:
                new_tags = [t for ns in ("genre", "theme") if ns in snap.tags for t in snap.tags[ns].tags_of(i)
                            if (ns, t) not in hist]
                if new_tags:
                    reasons.append(X.reason("fresh", tags=", ".join(new_tags[:3])))
        if reranker == "ga" and top_rel_unused:
            alt = sel.copy()
            alt[alt.tolist().index(pos)] = top_rel_unused[0]
            delta = base_fit - float(obj.fitness(alt[None, :])[0])
            comps["list_contribution_estimate"] = round(delta, 4)
            if delta > 0.005 and c["source"] != "popular_quality":
                reasons.append(X.reason("list_role", delta=round(delta, 3)))
        items.append({**card(snap, i), "score": round(float(relevance[pos]), 4), "reasons": reasons[:4],
                      "evidence_refs": [], "components": comps, "candidate_source": c["source"],
                      "constraints_passed": bool(elig_mask[i])})
    assert len({it["item_key"] for it in items}) == len(items)
    request_id = new_request_id()
    algorithm = f"{used_engine if mode != 'cold' else 'content' if prof is not None else 'popular_quality'}" \
                f"+{result.optimizer}"
    out = {"request_id": request_id, "items": items, "mode": mode,
           "cold_start": {"usable_ratings": n_usable, "warm_gate": gate,
                          "liked_tags": [list(t) for t in us.liked_tags]},
           "list_objective": {"fitness": round(result.fitness, 5), "summary": result.summary,
                              "greedy_baseline": greedy.summary, "available_terms": obj.available_terms(),
                              "weights": obj.weights, "optimizer": result.optimizer, "stopped": result.stopped,
                              "generations": len(result.log) - 1 if result.log else 0},
           "exploration": exploration(state, us, elig_mask, exclude={int(chosen[p]) for p in sel.tolist()}),
           "meta": _meta(state, k, len(items), algorithm, model_version, fallback, warnings)
           | {"candidate_pool": int(len(chosen)), "eligible": elig.count,
              "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    log_run(db, request_id, user, "home", {"k": k, "engine": engine, "reranker": reranker, "candidate_k": candidate_k},
            algorithm, model_version, state, k, items,
            [{"item_key": str(snap.keys[i]), "relevance": round(float(r), 4), "source": cands[int(i)]["source"]}
             for i, r in zip(chosen.tolist(), relevance.tolist(), strict=True)],
            {"mode": mode, "ga_log_tail": result.log[-3:] if result.log else [], "stopped": result.stopped})
    CACHE.put(key, out)
    return out


def _recall(state: ServingState, engine: str, us: UserState, rated_cf: list[int], elig_mask: np.ndarray):
    """Returns (catalogue ids, raw predicted rating, support) sorted desc, eligible only; None if unsupported."""
    if not rated_cf:
        return None
    cf_rated = np.array([state.cat_to_cf[i] for i in rated_cf])
    vals = np.array([us.ratings[i] for i in rated_cf])
    if engine == "icf":
        if state.icf is None:
            return None
        pred = state.icf.predict(cf_rated, vals, user_mean=us.rating_mean, n_ratings=len(us.ratings))
        ids = state.cf_items[pred.items]
        keep = elig_mask[ids]
        ids, p, s = ids[keep], pred.predicted[keep], pred.support[keep]
    elif engine == "mf":
        if state.mf is None:
            return None
        key = (us.user_id, us.profile_version, state.dataset_version)
        folded = _FOLD_IN.get(key)
        if folded is None:
            mf_cfg = state.settings.recommendation.get("mf", {})
            folded = state.mf.fold_in(cf_rated, vals, reg=float(mf_cfg.get("fold_in_reg", 1.0)),
                                      min_ratings=int(mf_cfg.get("fold_in_min_ratings", 3)))
            if folded is None:
                return None
            _FOLD_IN.put(key, folded)
        allp = np.clip(state.mf.predict_folded(*folded), 1, 10)
        ids = state.cf_items
        keep = elig_mask[ids]
        ids, p = ids[keep], allp[keep]
        s = np.zeros(len(ids))
    else:
        return None
    snap = state.snap
    order = np.lexsort((snap.keys[ids], -s, -p))
    return ids[order], p[order], s[order]


def exploration(state: ServingState, us: UserState, elig_mask: np.ndarray, exclude: set[int], k: int | None = None
                ) -> list[dict]:
    """Popular + well-rated titles, diversified by MMR over tags (clearly labelled exploration)."""
    snap = state.snap
    k = k or int(state.settings.recommendation.get("home", {}).get("exploration_k", 10))
    pool = np.array([i for i in np.flatnonzero(elig_mask).tolist() if i not in exclude])
    if not len(pool):
        return []
    pq = popular_quality(snap, pool)
    top = pool[np.lexsort((snap.keys[pool], -pq))[:60]]
    rel = popular_quality(snap, top)
    sim, valid = state.similarity["weighted_attributes"].pairwise(snap, top)
    from app.recommendation.rerank.ga import mmr

    obj = ListObjective(rel, sim, valid, None, None, {"relevance": 0.5, "diversity": 0.5})
    res = mmr(obj, k, lam=0.6)
    return [{**card(snap, int(top[p])), "reasons": [X.reason("popular_quality")],
             "candidate_source": "exploration_mmr"} for p in res.selected.tolist()]


def anime_rows(db: Session, keys: list[str]) -> dict[str, Anime]:
    from sqlalchemy import select

    return {a.item_key: a for a in db.scalars(select(Anime).where(Anime.item_key.in_(keys)))}
