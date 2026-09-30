"""Chat-to-find: intent parsing, session constraints, retrieval, grounded answers.

Every recommended title comes from the catalogue and passes the hard
constraints before any text is written. With no LLM configured (or on LLM
failure) the answer is a deterministic template over the same verified facts
and evidence ("local mode"), so the feature keeps working and never invents
titles, episode numbers or relations.
"""

from __future__ import annotations

import re
import time

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.conversation import constraint_chips, effective_intent, merge
from app.chat.grounded_answer import run_llm
from app.chat.intent import TagLexicon, parse
from app.chat.llm import get_llm
from app.core.errors import NotFound
from app.core.text import approx_tokens, norm
from app.db.models import Anime, ChatMessage, ChatSession, User, utcnow
from app.graph.relations import watch_order
from app.recommendation import explanations as X
from app.recommendation.constraints import ConstraintSet
from app.recommendation.features import stable_order
from app.retrieval.evidence import hybrid_search, item_docs, load_docs, snippet
from app.services import catalog
from app.services.recommend import _require, card, popular_quality
from app.services.serving import ServingState
from app.services.user_data import load_user_state

_LEXICONS: dict[str, TagLexicon] = {}


def lexicon(state: ServingState) -> TagLexicon:
    lx = _LEXICONS.get(state.dataset_version)
    if lx is None:
        lx = _LEXICONS[state.dataset_version] = TagLexicon(state.snap)
    return lx


# --------------------------------------------------------------------------- sessions


def _owned(db: Session, user: User, sid: int) -> ChatSession:
    s = db.get(ChatSession, sid)
    if s is None or s.user_id != user.id:
        raise NotFound("Chat not found.", code="chat_not_found")
    return s


def create_session(db: Session, user: User, title: str | None) -> ChatSession:
    s = ChatSession(user_id=user.id, title=(title or "New chat")[:120], state={})
    db.add(s)
    db.flush()
    return s


def list_sessions(db: Session, user: User) -> list[dict]:
    rows = db.scalars(select(ChatSession).where(ChatSession.user_id == user.id)
                      .order_by(ChatSession.updated_at.desc()).limit(200)).all()
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat(), "created_at": s.created_at.isoformat()}
            for s in rows]


def get_session(db: Session, user: User, sid: int) -> dict:
    s = _owned(db, user, sid)
    return {"id": s.id, "title": s.title, "state": {k: v for k, v in (s.state or {}).items() if k != "last_items"},
            "chips": constraint_chips(s.state or {}),
            "messages": [{"id": m.id, "role": m.role, "content": m.content, "payload": m.payload,
                          "created_at": m.created_at.isoformat()} for m in s.messages]}


def delete_session(db: Session, user: User, sid: int) -> None:
    db.delete(_owned(db, user, sid))


# --------------------------------------------------------------------------- helpers


def _constraint_set(st: dict) -> ConstraintSet:
    cs = ConstraintSet(min_score=st.get("min_score"), max_episodes=st.get("max_episodes"),
                       min_episodes=st.get("min_episodes"), types=st.get("types"),
                       include_tags=[tuple(t) for t in st.get("include_tags", [])],
                       exclude_tags=[tuple(t) for t in st.get("exclude_tags", [])])
    for rid in cs.active_rules():
        if rid.startswith("filter."):
            cs.sources[rid] = "your chat request"
    return cs


def _facts(state: ServingState, db: Session, keys: list[str]) -> list[dict]:
    snap = state.snap
    rows = {a.item_key: a for a in db.scalars(select(Anime).where(Anime.item_key.in_(keys)))}
    out = []
    for k in keys:
        i = snap.index[k]
        a = rows.get(k)
        c = card(snap, i)
        out.append({"item_key": k, "title": c["title"], "type": c["type"], "episodes": c["episodes"],
                    "catalog_score": c["catalog_score"], "genres": c["genres"],
                    "themes": snap.tags["theme"].tags_of(i) if "theme" in snap.tags else [],
                    "studios": snap.tags["studio"].tags_of(i) if "studio" in snap.tags else [],
                    "aired": a.aired_raw if a else None, "status": a.status if a else None,
                    "duration": a.duration_raw if a else None})
    return out


def _soft_adjust(state: ServingState, st: dict, cand: np.ndarray, ref: int | None) -> tuple[np.ndarray, dict]:
    snap = state.snap
    adj = np.zeros(len(cand))
    notes: dict[int, list[str]] = {}
    eps = snap.episodes[cand]
    prefs = st.get("soft_preferences", [])

    def note(pos, text):
        notes.setdefault(int(pos), []).append(text)

    if "shorter_than_reference" in prefs and ref is not None and np.isfinite(snap.episodes[ref]):
        shorter = np.isfinite(eps) & (eps < snap.episodes[ref])
        adj += np.where(shorter, 0.15, -0.1)
        for p in np.flatnonzero(shorter):
            note(p, f"Shorter than {snap.titles[ref]} ({int(eps[p])} vs {int(snap.episodes[ref])} episodes)")
    if "longer_than_reference" in prefs and ref is not None and np.isfinite(snap.episodes[ref]):
        adj += np.where(np.isfinite(eps) & (eps > snap.episodes[ref]), 0.15, -0.1)
    if "short" in prefs:
        short = np.isfinite(eps) & (eps <= 13)
        adj += np.where(short, 0.1, 0.0)
        for p in np.flatnonzero(short):
            note(p, f"Short ({int(eps[p])} episodes)")
    if "long" in prefs:
        adj += np.where(np.isfinite(eps) & (eps >= 24), 0.1, 0.0)
    for sign, pairs in ((1.0, st.get("soft_tags", [])), (-1.0, st.get("soft_avoid_tags", []))):
        for ns, v in pairs:
            tm = snap.tags.get(ns)
            col = tm.col.get(norm(v)) if tm is not None else None
            if col is None:
                continue
            has = np.asarray(tm.X[cand][:, col].toarray()).ravel() > 0
            adj += sign * np.where(has, 0.1, 0.0)
            if sign > 0:
                for p in np.flatnonzero(has):
                    note(p, f"Has {v} (your preference)")
    counts = snap.train_interactions
    if counts is not None and counts.max() > 0 and ("obscure" in prefs or "popular" in prefs):
        nov = 1 - np.log1p(counts[cand]) / np.log1p(counts.max())
        adj += 0.1 * (nov if "obscure" in prefs else 1 - nov)
    if "high_score" in prefs:
        adj += 0.1 * np.nan_to_num(snap.score[cand]) / 10
    return adj, notes


def _user_filters(cs: ConstraintSet) -> bool:
    return any(r.startswith("filter.") for r in cs.active_rules())


def _relax(state: ServingState, cs: ConstraintSet, k: int = 5) -> list[dict]:
    return state.constraints.suggest_relaxations(cs, k)


# --------------------------------------------------------------------------- handlers


def _h_fact(state, db, st, pq, msg) -> dict:
    ref = st.get("reference")
    if not ref:
        return {"clarification": "Which title do you mean? Tell me its name (English or romaji).", "items": []}
    detail = catalog.detail(state, db, None, ref["item_key"])
    low = msg.lower()
    parts = [f"{detail['title']}"]
    if detail.get("title_english") and detail["title_english"] != detail["title"]:
        parts[0] += f" ({detail['title_english']})"
    unknowns = []
    t = detail.get("type") or "title"
    eps = detail.get("episodes")
    parts.append(f"is a {t}" + (f" with {eps} episodes" if eps else " with an unknown episode count"))
    if not eps:
        unknowns.append("episode count")
    if detail.get("duration_raw"):
        parts.append(f"({detail['duration_raw']})")
    text = " ".join(parts) + "."
    if detail.get("status"):
        text += f" Status: {detail['status']}."
    if detail.get("aired"):
        text += f" Aired: {detail['aired']}."
    studios = detail["tags"].get("studio", [])
    text += f" Studio: {', '.join(studios)}." if studios else " Studio: unknown."
    if not studios:
        unknowns.append("studio")
    genres = detail["tags"].get("genre", [])
    if genres:
        text += f" Genres: {', '.join(genres)}."
    if detail.get("catalog_score") is not None:
        label = " (MAL 2023 snapshot)" if ref["item_key"].startswith("mal:") else ""
        text += f" Catalogue score: {detail['catalog_score']}{label}."
    else:
        text += " It has no catalogue score."
        unknowns.append("score")
    if re.search(r"voice|seiyuu|director|composer|who (?:voices|directed)", low):
        unknowns.append("staff / cast details are not in the available data")
        text += " Staff and voice-cast details are not in the available data."
    docs = item_docs(db, ref["item_key"], ["attributes", "synopsis", "relations"])[:3]
    return {"answer": text, "items": [ref["item_key"]], "evidence": docs, "unknowns": unknowns,
            "item_reasons": {ref["item_key"]: [{"code": "resolved_title", "params": {},
                                                "text": f"Matched \"{ref['mention']}\" ({ref['match']})"}]}}


def _rank_output(state, keys: list[str], reasons: dict, docs: list[dict], intro: str, unknowns=None) -> dict:
    return {"items": keys, "item_reasons": reasons, "evidence": docs, "answer_intro": intro,
            "unknowns": unknowns or []}


def _h_similar(state, db, st, pq, user) -> dict:
    snap = state.snap
    ref = st.get("reference")
    if not ref:
        return {"clarification": "Which title should the suggestions be similar to?", "items": []}
    r = snap.index[ref["item_key"]]
    cs = _constraint_set(st)
    cs.exclude_items = {r}
    us = load_user_state(db, user, snap)
    cs.exclude_items |= us.items_with_status(["completed"])
    if us.excluded_items:
        cs.user_excluded_items = set(us.excluded_items)
    if us.excluded_tags:
        cs.user_excluded_tags = list(us.excluded_tags)
    elig = state.constraints.evaluate(cs)
    cand = np.flatnonzero(elig.mask)
    if not len(cand):
        return {"items": [], "relaxation_options": _relax(state, cs),
                "answer": f"No titles similar to {ref['title']} meet all of your conditions.", "no_match": True}
    scorer = state.similarity.get("tag_semantic_hybrid") or state.similarity["weighted_attributes"]
    res = scorer.score(snap, r, cand)
    ok = np.isfinite(res.score)
    base = np.where(ok, res.score, -np.inf)
    adj, notes = _soft_adjust(state, st, cand, r)
    final = base + adj
    order = stable_order(final, snap.keys[cand], res.coverage)[:5]
    order = [p for p in order if np.isfinite(final[p])]
    keys, reasons = [], {}
    for p in order:
        i = int(cand[p])
        k = str(snap.keys[i])
        keys.append(k)
        rs = X.similarity_reasons(snap, r, i, res.available, res.sims, int(p))
        rs = [x for x in rs if x["code"] != "missing_dims"][:3]
        rs += [{"code": "soft_preference", "params": {}, "text": t} for t in notes.get(int(p), [])[:2]]
        reasons[k] = rs
    docs = []
    for k in keys:
        docs += item_docs(db, k, ["attributes"])[:1]
    docs += item_docs(db, ref["item_key"], ["attributes"])[:1]
    return _rank_output(state, keys, reasons, docs, f"Titles similar to {ref['title']}"
                        + (" that meet your conditions" if _user_filters(cs) else "") + ":")


def _h_recommend(state, db, st, pq, user) -> dict:
    snap = state.snap
    cs = _constraint_set(st)
    us = load_user_state(db, user, snap)
    cs.exclude_items = us.items_with_status(["completed"])
    if us.excluded_items:
        cs.user_excluded_items = set(us.excluded_items)
    if us.excluded_tags:
        cs.user_excluded_tags = list(us.excluded_tags)
    elig = state.constraints.evaluate(cs)
    cand = np.flatnonzero(elig.mask)
    if not len(cand):
        return {"items": [], "relaxation_options": _relax(state, cs), "no_match": True,
                "answer": "No titles meet all of those conditions."}
    free = pq.free_text
    tag_words = {norm_w for t in st.get("include_tags", []) + st.get("exclude_tags", []) for norm_w in t[1].lower().split()}
    content_words = [w for w in re.findall(r"[a-zA-Z']+", free.lower()) if len(w) > 2 and w not in tag_words]
    descriptive = len(content_words) >= 3 and pq.language == "en"
    reasons: dict[str, list] = {}
    if descriptive:
        eligible_keys = set(snap.keys[cand].tolist())
        rr = hybrid_search(state, db, free, eligible_keys, kinds=["synopsis", "attributes", "review", "characters"],
                           item_k=5)
        keys = [it.item_key for it in rr.items]
        if not keys:
            return {"items": [], "answer": "I could not match that description to any title. Could you add details "
                    "such as the setting, a character, or the genre?", "low_evidence": True,
                    "degraded": rr.degraded}
        doc_ids = [d.doc_id for it in rr.items for d in it.docs]
        docs_map = load_docs(db, doc_ids)
        docs = [docs_map[d] for d in doc_ids if d in docs_map]
        for it in rr.items:
            kinds = sorted({d.kind for d in it.docs})
            reasons[it.item_key] = [X.reason("retrieved", kind=" and ".join(kinds))]
        out = _rank_output(state, keys, reasons, docs, "Titles that match your description"
                           + (" and conditions" if _user_filters(cs) else "") + ":")
        out["retrieval"] = {"channels": rr.channels, "degraded": rr.degraded}
        return out
    pq_scores = popular_quality(snap, cand)
    adj, notes = _soft_adjust(state, st, cand, None)
    final = pq_scores + adj
    order = stable_order(final, snap.keys[cand])[:5]
    keys = []
    for p in order:
        k = str(snap.keys[int(cand[p])])
        keys.append(k)
        reasons[k] = [X.reason("popular_quality")] + [{"code": "soft_preference", "params": {}, "text": t}
                                                      for t in notes.get(int(p), [])[:2]]
    docs = []
    for k in keys:
        docs += item_docs(db, k, ["attributes"])[:1]
    intro = "Well-rated, popular titles that meet your conditions:" if _user_filters(cs) else \
        "Some popular, well-rated titles:"
    return _rank_output(state, keys, reasons, docs, intro)


def _h_relation(state, db, st, pq) -> dict:
    ref = st.get("reference")
    if not ref:
        return {"clarification": "Which title's sequels or watch order do you want to know?", "items": []}
    wo = watch_order(db, ref["item_key"])
    rels = catalog.relations(state, db, ref["item_key"])
    docs = item_docs(db, ref["item_key"], ["relations"])
    if not rels and not wo["watch_first"]:
        cap = state.cap("relations")
        return {"items": [ref["item_key"]], "evidence": [], "unknowns": ["relations"],
                "answer": f"I have no confirmed relation records for {ref['title']}. That does not mean it has no "
                          f"sequel or prequel - the relation data is incomplete ({cap.detail})"}
    lines = []
    if wo["watch_first"]:
        lines.append("Watch first: " + ", ".join(x["title"] for x in wo["watch_first"]) + ".")
    if wo["direct_sequels"]:
        lines.append("Direct sequel(s): " + ", ".join(x["title"] for x in wo["direct_sequels"]) + ".")
    other = [r for r in rels if r["relation_type"] not in ("sequel", "prequel")]
    if other:
        lines.append("Other relations: " + "; ".join(f"{r['relation_type'].replace('_', ' ')}: {r['to_title']}"
                                                      for r in other[:6]) + ".")
    keys = [ref["item_key"]] + [x["item_key"] for x in wo["watch_first"] + wo["direct_sequels"]
                                if x["item_key"] in state.snap.index]
    return {"items": list(dict.fromkeys(keys)), "evidence": docs,
            "answer": f"{ref['title']}: " + " ".join(lines) + " (Source: " + "; ".join(wo["sources"][:2]) + ")"
            if wo["sources"] else f"{ref['title']}: " + " ".join(lines)}


def _h_scene(state, db, st, pq) -> dict:
    ref = st.get("reference")
    cap = state.cap("scenes")
    if cap.status == "limited_by_data":
        base = ("I don't have verified scene or episode data, so I can't tell you which episode a scene is in.")
        if ref:
            return {"items": [ref["item_key"]], "unknowns": ["episode"], "answer": base + f" {ref['title']} is in "
                    "the catalogue, but its scenes are not.", "evidence": item_docs(db, ref["item_key"], ["synopsis"])[:1]}
        rr = hybrid_search(state, db, pq.free_text or "", None, kinds=["synopsis"], item_k=3)
        keys = [it.item_key for it in rr.items]
        docs = load_docs(db, [d.doc_id for it in rr.items for d in it.docs])
        return {"items": keys, "unknowns": ["episode"], "evidence": list(docs.values()),
                "answer": base + (" These titles' synopses match your description at the plot level:" if keys else
                                  " Could you describe the story or characters instead?"),
                "item_reasons": {k: [X.reason("retrieved", kind="synopsis")] for k in keys}}
    eligible = {ref["item_key"]} if ref else None
    rr = hybrid_search(state, db, pq.free_text or "", eligible, kinds=["scene"], item_k=3, evidence_per_item=1)
    if not rr.items:
        return {"items": [ref["item_key"]] if ref else [], "unknowns": ["episode"],
                "answer": ("I found no verified scene matching that description"
                           + (f" in {ref['title']}" if ref else "") + ". I won't guess an episode number.")}
    docs = load_docs(db, [d.doc_id for it in rr.items for d in it.docs])
    top = docs[rr.items[0].docs[0].doc_id]
    ep = top["locator"].get("episode")
    verified = top["source_type"] == "scene_verified"
    title = state.snap.titles[state.snap.index[top["item_key"]]]
    if verified and ep:
        ts = top["locator"].get("timestamp_seconds")
        when = f", around {ts // 60}:{ts % 60:02d}" if ts else ""
        text = f"That sounds like {title}, episode {ep}{when} (verified scene record)."
        unknowns = [] if ts else ["timestamp"]
    else:
        text = f"That sounds like a scene from {title}, but the episode is not recorded in the data."
        unknowns = ["episode"]
    return {"items": [it.item_key for it in rr.items], "evidence": list(docs.values()), "unknowns": unknowns,
            "answer": text, "spoiler": True,
            "item_reasons": {it.item_key: [X.reason("retrieved", kind="scene records")] for it in rr.items}}


def _h_plan(state, db, st, pq, user) -> dict:
    from app.services import plans

    budget = pq.budget_minutes or st.get("budget_minutes") or int(state.settings.planner.get("default_budget_minutes", 900))
    req = {"budget_minutes": int(budget), "max_titles": int(pq.max_titles or 5), "mode": "complete",
           "diversity": pq.diversity or "normal",
           "include_genres": [v for ns, v in st.get("include_tags", []) if ns == "genre"],
           "exclude_genres": [v for ns, v in st.get("exclude_tags", []) if ns == "genre"],
           "min_score": st.get("min_score"), "max_episodes_per_title": st.get("max_episodes"),
           "allow_prerequisites": True, "daily_slots": [], "must_include": [], "break_minutes_per_title": 0}
    plan = plans.compute(state, db, user, req)
    hours = budget / 60
    understood = (f"I read that as {int(budget)} minutes ({hours:g} h), up to {req['max_titles']} titles"
                  + (", with varied styles" if req["diversity"] == "high" else "") + ", watching each title in full.")
    if not plan["items"]:
        return {"items": [], "plan_preview": plan, "answer": understood + " No title fits that budget and those "
                "conditions. Try a larger budget or fewer filters on the Plan page."}
    lines = [f"{n}. {it['title']} - {it['minutes']:.0f} min" + (" (watch first)" if it["prerequisite_of"] else "")
             for n, it in enumerate(plan["items"], start=1)]
    text = (understood + " Here is a plan:\n" + "\n".join(lines)
            + f"\nTotal {plan['total_minutes']:.0f} min, {plan['remaining_minutes']:.0f} min left. "
              "Open it on the Plan page to switch to partial episodes, add daily slots, save or export.")
    return {"items": [it["item_key"] for it in plan["items"]], "plan_preview": plan, "answer": text,
            "item_reasons": {it["item_key"]: it["reasons"] for it in plan["items"]}}


def _h_screenshot(state) -> dict:
    cap = state.cap("screenshot")
    if cap.status == "limited_by_data":
        return {"items": [], "answer": "Screenshot identification isn't available: there are no frame-level images "
                "in the data (cover art is not used as a substitute). Describe the scene in words instead."}
    return {"items": [], "answer": "Upload the screenshot with the image button next to the message box. "
            + ("Note: the current gallery is a synthetic fixture for pipeline testing." if cap.status == "degraded" else "")}


# --------------------------------------------------------------------------- main entry


def post_message(state: ServingState, db: Session, user: User, sid: int, message: str,
                 save_preferences: bool = False) -> dict:
    t0 = time.perf_counter()
    _require(state)
    sess = _owned(db, user, sid)
    pq = parse(message, state.snap, state.titles, lexicon(state))
    st, changes, refinement = merge(sess.state or {}, pq, message)
    intent = effective_intent(pq, st, refinement)
    if pq.budget_minutes:
        st["budget_minutes"] = pq.budget_minutes
    warnings = list(pq.notes)
    result: dict
    if pq.needs_clarification:
        result = {"items": [], "clarification": pq.clarification}
    elif intent in ("anime_fact", "title_lookup"):
        result = _h_fact(state, db, st, pq, message)
    elif intent == "similar_to":
        result = _h_similar(state, db, st, pq, user)
    elif intent == "relation_query":
        result = _h_relation(state, db, st, pq)
    elif intent == "scene_lookup":
        result = _h_scene(state, db, st, pq)
    elif intent == "viewing_plan":
        result = _h_plan(state, db, st, pq, user)
    elif intent == "screenshot_lookup":
        result = _h_screenshot(state)
    elif intent == "out_of_scope":
        result = {"items": [], "answer": "I can help you find anime, answer questions about titles in the catalogue, "
                  "and plan what to watch. Try: \"a short mystery with at most 12 episodes\"."}
    else:
        result = _h_recommend(state, db, st, pq, user)
    if result.get("degraded"):
        warnings += result["degraded"]
    if result.get("retrieval", {}).get("degraded"):
        warnings += result["retrieval"]["degraded"]

    keys = [k for k in result.get("items", []) if k in state.snap.index]
    evidence = result.get("evidence") or []
    template = _template(result, keys, state)
    answer_mode, answer_text, llm_meta, validation, claims = "local_template", template, [], None, []
    unknowns = list(result.get("unknowns", []))
    llm = get_llm(state.settings)
    if llm is not None and keys and intent in ("similar_to", "preference_recommendation", "anime_fact",
                                               "relation_query", "scene_lookup"):
        facts = _facts(state, db, keys)
        constraints_public = {k: st.get(k) for k in ("include_tags", "exclude_tags", "max_episodes", "min_episodes",
                                                     "min_score", "types", "soft_preferences")}
        clean, calls, validation = run_llm(llm, message, intent, constraints_public, facts, evidence,
                                           int(state.settings.llm.max_calls_per_request))
        llm_meta = [c.public() for c in calls]
        if clean is not None:
            answer_mode, answer_text, claims = "llm", clean["answer"], clean["claims"]
            unknowns = list(dict.fromkeys(unknowns + clean["unknowns"]))
        else:
            warnings.append("The language model was unavailable or its answer failed validation; "
                            "showing the local evidence-based answer.")
    elif llm is None and state.cap("llm").status != "ready":
        pass
    items_out = []
    reasons = result.get("item_reasons", {})
    ev_by_item: dict[str, list[str]] = {}
    for d in evidence:
        ev_by_item.setdefault(d["item_key"], []).append(d["doc_id"])
    for k in keys:
        c = card(state.snap, state.snap.index[k])
        items_out.append({**c, "reasons": reasons.get(k, []), "evidence_refs": ev_by_item.get(k, [])[:3],
                          "constraints_passed": True})
    evidence_out = [{"doc_id": d["doc_id"], "item_key": d["item_key"], "title": d["title"], "kind": d["kind"],
                     "source_type": d["source_type"], "source_ref": d["source_ref"], "source_url": d.get("source_url"),
                     "spoiler_level": d.get("spoiler_level"), "locator": d.get("locator", {}),
                     "snippet": snippet(d["body"])} for d in evidence]
    st["last_intent"] = intent
    st["last_items"] = keys
    payload = {"intent": intent, "parser": pq.parser, "answer": answer_text, "answer_mode": answer_mode,
               "items": items_out, "evidence": evidence_out, "claims": claims, "unknowns": unknowns,
               "clarification": result.get("clarification"), "relaxation_options": result.get("relaxation_options", []),
               "plan_preview": result.get("plan_preview"), "spoiler": bool(result.get("spoiler")),
               "chips": constraint_chips(st), "changes": changes, "refinement": refinement,
               "warnings": warnings, "parsed": pq.public(),
               "meta": {"dataset_mode": state.dataset_mode, "dataset_version": state.dataset_version,
                        "llm": llm_meta, "llm_capability": state.cap("llm").status,
                        "validation": validation.public() if validation else None,
                        "retrieval": result.get("retrieval"),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    db.add(ChatMessage(session_id=sess.id, role="user", content=message, parsed=pq.public(),
                       dataset_version=state.dataset_version))
    db.add(ChatMessage(session_id=sess.id, role="assistant", content=answer_text, payload=payload,
                       model_info={"answer_mode": answer_mode, "calls": llm_meta},
                       dataset_version=state.dataset_version))
    sess.state = st
    if sess.title == "New chat":
        sess.title = message[:60]
    sess.updated_at = utcnow()
    if save_preferences and (st.get("include_tags") or st.get("exclude_tags")):
        _save_prefs(db, user, st)
        payload["saved_preferences"] = True
    db.flush()
    return {"session_id": sess.id, **payload}


def _template(result: dict, keys: list[str], state: ServingState) -> str:
    if result.get("clarification"):
        return result["clarification"]
    if result.get("answer"):
        return result["answer"]
    intro = result.get("answer_intro") or "Here is what I found:"
    lines = [intro]
    reasons = result.get("item_reasons", {})
    for n, k in enumerate(keys, start=1):
        i = state.snap.index[k]
        eps = state.snap.episodes[i]
        meta = f"{int(eps)} ep" if np.isfinite(eps) else "episodes unknown"
        why = "; ".join(r["text"] for r in reasons.get(k, [])[:2])
        lines.append(f"{n}. {state.snap.titles[i]} ({meta})" + (f" - {why}" if why else ""))
    if result.get("unknowns"):
        lines.append("Not in the available data: " + ", ".join(result["unknowns"]) + ".")
    return "\n".join(lines)


def _save_prefs(db: Session, user: User, st: dict) -> None:
    from app.services import user_data

    pref = user_data.get_preferences(db, user)
    liked = []
    if pref.liked_tag_ids:
        from app.db.models import Tag

        liked = [(t.namespace, t.value) for t in db.scalars(select(Tag).where(Tag.id.in_(pref.liked_tag_ids)))]
    liked = list(dict.fromkeys(liked + [tuple(t) for t in st.get("include_tags", [])]))
    user_data.update_preferences(db, user, liked, None, None)
    for t in st.get("exclude_tags", []):
        user_data.add_exclusion(db, user, "tag", tag=tuple(t), source="chat_saved")


def token_estimate(text: str) -> int:
    return approx_tokens(text)
