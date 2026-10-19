"""Chat-to-find evaluation (algorithm 4).

Query sets (each reported separately):
  labelled         data/evaluation/rag_queries.jsonl, named by the rows' label_source (e.g. the team package's
                   queries, split by category: scene / semantic / constraint / unanswerable) or the synthetic
                   generator's queries
  auto_titles      REAL mode, auto-generated: exact / alias / one-typo title lookups (ground truth by construction)
  auto_constraints REAL mode, auto-generated: "a <genre> anime with at most N episodes, no <genre2>"
  auto_relations   REAL mode, auto-generated from confirmed sequel relations (a seeded sample of 60 titles; any
                   direct sequel counts as relevant)
  auto_no_episode  REAL mode: "which episode of <title> has the ... scene" (no scene data exists: an
                   episode number in the answer is a fabrication)
  auto_scene_outside   REAL mode, with scene records: "which anime has a story where <first synopsis sentence>?"
                   for popular titles WITHOUT scene records. Naming a verified scene is always wrong here; the
                   verbatim sentence makes Hits@5 an upper bound for plot-level matching
  auto_scene_mismatch  REAL mode, with scene records: a title that HAS scene records, asked about a scene none of
                   them describes ("... file their taxes at a post office"): no scene may be claimed
Auto-generated sets are NOT human-annotated relevance judgements; they test robustness, constraint
compliance and refusal behaviour.

Retrieval ablation: labelled descriptive queries (lexical, dense and hybrid RRF item-level Hits@5 /
MRR@10 over all served titles) and labelled scene queries (the same three channels over the scene corpus,
the synopsis corpus and both, with scene attribution where the label names the scene).
Plain LLM vs RAG needs a configured model; without credentials it is reported as blocked, never faked.
Task success (TSR) by expected outcome:
  matches                   at least one relevant title in the top 5, every returned title satisfies the
                            hard constraints, no fabricated episode number
  no_match                  no titles returned and no fabricated episode number
  insufficient_information  no titles returned (a referenced title may be shown) and no invented specifics
                            (episode numbers, timecodes, dates, prices)
  (rows without an outcome) answerable -> as "matches"; unanswerable -> no fabricated episode; a
                            contradictory request -> a clarification is asked
Scene attribution (rows naming the expected scene): scene@3 = the expected scene record is among the
answer's evidence; scene@1 = it is the first scene record cited.
"""

from __future__ import annotations

import re
import time

import numpy as np
from sqlalchemy import select

from app.retrieval.evidence import hybrid_search
from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci, hits_at_k, mrr_at_k
from evaluation.sandbox import make_user, sandbox_settings
from pipelines.common import read_jsonl

EPISODE_CLAIM = re.compile(r"\bepisode\s*(?:#|no\.?\s*)?\d+\b|第\s*\d+\s*集", re.I)
SCENE_CLAIM = re.compile(r"\(verified scene record\)|That sounds like a scene from")
TIMECODE_CLAIM = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
DATE_CLAIM = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? "
                        r"\d{1,2},? (?:19|20)\d{2}\b", re.I)
MONEY_CLAIM = re.compile(r"[$€£¥]\s*\d|\b\d[\d,.]*\s*(?:usd|eur|jpy|gbp|dollars?|yen|euros?)\b", re.I)
TAG_KEYS = {"genre": "genres", "theme": "themes", "demographic": "demographics", "studio": "studios"}


def _compliant(snap, key: str, hc: dict) -> bool:
    """Every hard constraint of the label holds for the title; an unknown value never counts as satisfying one."""
    from app.core.text import norm

    i = snap.index[key]
    ep, sc, yr = snap.episodes[i], snap.score[i], snap.year[i]
    for name, v, upper in (("max_episodes", ep, True), ("min_episodes", ep, False), ("max_score", sc, True),
                           ("min_score", sc, False), ("year_to", yr, True), ("year_from", yr, False)):
        lim = hc.get(name)
        if lim is not None and not (np.isfinite(v) and (v <= lim if upper else v >= lim)):
            return False
    if hc.get("types") and (snap.type[i] or "").casefold() not in {t.casefold() for t in hc["types"]}:
        return False
    for ns, plural in TAG_KEYS.items():
        tags = {norm(t) for t in snap.tags[ns].tags_of(i)} if ns in snap.tags else set()
        if any(norm(g) not in tags for g in hc.get(f"include_{plural}", [])):
            return False
        if any(norm(g) in tags for g in hc.get(f"exclude_{plural}", [])):
            return False
    return True


_EPISODE_TITLES: dict[str, list[str]] = {}


def _episode_claim(answer: str, evidence: list[dict], snap) -> bool:
    """An episode number in the answer without a verified scene record. Catalogue titles that themselves contain
    "Episode N" ("Full Metal Panic! The Second Raid Episode 00") are quoted names, not claims, and are removed first."""
    titles = _EPISODE_TITLES.get(snap.dataset_version)
    if titles is None:
        names = {t for t in list(snap.titles) + list(snap.titles_english) if t and EPISODE_CLAIM.search(t)}
        titles = _EPISODE_TITLES[snap.dataset_version] = sorted(names, key=len, reverse=True)
    text = answer
    for t in titles:
        if t in text:
            text = text.replace(t, " ")
    return bool(EPISODE_CLAIM.search(text)) and not any(e["source_type"] == "scene_verified" for e in evidence)


def _invented_specifics(answer: str, evidence: list[dict], snap) -> bool:
    """An episode number without a verified scene record, or a timecode / date / price in the answer."""
    ep = _episode_claim(answer, evidence, snap)
    return ep or bool(TIMECODE_CLAIM.search(answer) or DATE_CLAIM.search(answer) or MONEY_CLAIM.search(answer))


def _scene_ids(evidence: list[dict]) -> list[str]:
    return [e["locator"].get("scene_id") for e in evidence if e.get("kind") == "scene" and e.get("locator")]


def _auto_sets(state, snap, rng, n=120) -> dict[str, list[dict]]:
    from app.core.text import norm

    rec = np.flatnonzero(snap.recommendable & snap.in_catalog)
    pop = rec[np.argsort(-np.nan_to_num(snap.members[rec]))][:2000]
    picks = rng.choice(pop, size=min(n, len(pop)), replace=False)
    titles = []
    for i in picks:
        t = snap.titles[i]
        words = t.split()
        w = max(range(len(words)), key=lambda k: len(words[k]))
        typo_word = words[w][:1] + words[w][2:] if len(words[w]) > 4 else words[w]
        typo = " ".join(words[:w] + [typo_word] + words[w + 1:])
        variants = [("exact", t), ("typo", typo)]
        if snap.titles_english[i] and snap.titles_english[i] != t:
            variants.append(("english_alias", snap.titles_english[i]))
        for kind, text in variants:
            titles.append({"query_id": f"title-{kind}-{i}", "query": f"tell me about {text}", "intent": "anime_fact",
                           "relevant_item_keys": [str(snap.keys[i])], "hard_constraints": {}, "answerable": True,
                           "variant": kind})
    gm = snap.tags.get("genre")
    # genres of served, recommendable titles only; content-rule genres (e.g. Hentai) are never served, so a
    # request for them has no answer by design and is not a constraint-compliance case
    excluded = {norm(g) for g in state.settings.content_rules.exclude_genres}
    counts = np.asarray(gm.X[rec].sum(axis=0)).ravel() if gm is not None else np.zeros(0)
    genres = [g for g in gm.vocab if counts[gm.col[norm(g)]] > 50 and norm(g) not in excluded] if gm is not None else []
    cons = []
    for k in range(min(40, len(genres) * 3)):
        g = genres[k % len(genres)]
        g2 = genres[(k * 7 + 3) % len(genres)]
        if g2 == g:
            continue
        n_ep = int(rng.choice([6, 12, 13, 24, 26]))
        cons.append({"query_id": f"cons-{k}", "query": f"a {g.lower()} anime with at most {n_ep} episodes and no {g2.lower()}",
                     "intent": "preference_recommendation", "relevant_item_keys": None,
                     "hard_constraints": {"include_genres": [g], "exclude_genres": [g2], "max_episodes": n_ep},
                     "answerable": True})
    rels = []
    from app.db.models import Relation
    from app.db.session import session_scope

    with session_scope(state.settings.db_url) as db:
        rows = db.query(Relation.from_item, Relation.to_item).filter(Relation.relation_type == "sequel",
                                                                     Relation.to_item.is_not(None)).all()
    by_src: dict[str, set[str]] = {}
    for a, b in rows:
        if a in snap.index and b in snap.index:
            by_src.setdefault(a, set()).add(b)
    # a seeded sample over every title with a confirmed, served sequel; any direct sequel answers the question
    srcs = sorted(by_src)
    for a in sorted(rng.choice(srcs, size=min(60, len(srcs)), replace=False).tolist()) if srcs else []:
        rels.append({"query_id": f"rel-{a}", "query": f"what is the sequel to {snap.titles[snap.index[a]]}?",
                     "intent": "relation_query", "relevant_item_keys": sorted(by_src[a]), "hard_constraints": {},
                     "answerable": True})
    noep = [{"query_id": f"noep-{i}", "query": f"which episode of {snap.titles[i]} has the scene where they fight "
             "on the rooftop in the rain?", "intent": "scene_lookup", "relevant_item_keys": [], "hard_constraints": {},
             "answerable": False} for i in picks[:30]]
    out = {"auto_titles": titles, "auto_constraints": cons, "auto_relations": rels[:60], "auto_no_episode": noep}
    out.update(_auto_scene_sets(state, snap, pop))
    return out


def _auto_scene_sets(state, snap, pop) -> dict[str, list[dict]]:
    """Scene questions the scene layer must NOT answer with a verified scene (see the module docstring)."""
    from app.db.models import Anime, Scene
    from app.db.session import session_scope

    with session_scope(state.settings.db_url) as db:
        with_scenes = sorted(set(db.scalars(select(Scene.item_key)).all()))
        if not with_scenes:
            return {}
        outside = []
        for i in pop:
            key = str(snap.keys[i])
            if key in with_scenes:
                continue
            a = db.get(Anime, key)
            sents = re.split(r"(?<=[.!?])\s+", (a.synopsis or "").strip()) if a else []
            sent = next((x for x in sents if 12 <= len(x.split()) <= 40 and "(Source" not in x), None)
            if sent:
                outside.append({"query_id": f"scene-out-{key}", "query": "Which anime has a story where "
                                + sent[0].lower() + sent[1:].rstrip(".!?") + "?", "intent": "scene_lookup",
                                "relevant_item_keys": [key], "hard_constraints": {}, "answerable": True,
                                "no_scene_claim": True})
            if len(outside) >= 60:
                break
    mismatch = [{"query_id": f"scene-mismatch-{k}", "query": f"which episode of {snap.titles[snap.index[k]]} has the "
                 "scene where they file their taxes at a post office?", "intent": "scene_lookup",
                 "relevant_item_keys": [], "hard_constraints": {}, "answerable": False, "no_scene_claim": True}
                for k in with_scenes[:40] if k in snap.index]
    return {"auto_scene_outside": outside, "auto_scene_mismatch": mismatch}


def _labelled(settings) -> tuple[str, list[dict]] | None:
    """The labelled query file of the mode, named by where its labels come from (never assumed to be human).

    data/evaluation/rag_queries.jsonl labels the real catalogue (MAL item keys), so it is used in real mode only;
    synthetic mode uses the generator's queries for the synthetic world."""
    if settings.mode == "synthetic":
        p = settings.synthetic_dir / "evaluation" / "rag_queries.jsonl"
        return ("labelled_synthetic_generator", list(read_jsonl(p))) if p.exists() else None
    path = settings.data_root / "evaluation" / "rag_queries.jsonl"
    if path.exists():
        rows = list(read_jsonl(path))
        sources = {r.get("label_source") for r in rows}
        name = str(sources.pop()) if len(sources) == 1 and None not in sources else "labelled_unmarked_source"
        return name, rows
    return None


def _split_by_category(name: str, rows: list[dict]) -> dict[str, list[dict]]:
    if not any(r.get("category") for r in rows):
        return {name: rows}
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(f"{name}:{r.get('category') or 'other'}", []).append(r)
    return out


def _task_success(q: dict, out: dict, keys: list[str], rel: set[str], compliant: bool, fabricated_ep: bool,
                  m: dict, snap) -> float:
    outcome = q.get("expected_outcome")
    if outcome == "no_match":
        return float(not keys and not fabricated_ep)
    if outcome == "insufficient_information":
        shown = set(keys) - {r["item_key"] for r in (out.get("parsed") or {}).get("reference_titles", [])}
        return float(not shown and not _invented_specifics(out["answer"], out["evidence"], snap))
    if q.get("no_scene_claim") and SCENE_CLAIM.search(out["answer"]):
        return 0.0
    if q.get("answerable") is False:
        hc = q.get("hard_constraints") or {}
        if hc.get("min_episodes") is not None and "max_episodes" in hc:
            return float(bool(out.get("clarification")))
        return float(not fabricated_ep)
    if rel:
        return float(m["hits@5"] == 1.0 and compliant and not fabricated_ep)
    return float(bool(keys) and compliant and not fabricated_ep)


def _ablation(state, db, queries: list[dict], allowed: set[str], corpora: list[tuple[str, list[str] | None]]) -> list:
    """Lexical / dense / hybrid item-level retrieval on the raw query text, per corpus."""
    out = []
    saved = state.dense
    for corpus, kinds in corpora:
        for variant in ("lexical", "dense", "hybrid"):
            if variant == "dense" and saved is None:
                continue
            h1, h5, mr, s1, s3 = [], [], [], [], []
            state.dense = None if variant == "lexical" else saved
            try:
                for q in queries:
                    if variant == "dense":
                        hits = saved.search(q["query"], allowed, top_n=50, kinds=kinds)
                        items = list(dict.fromkeys(h.item_key for h in hits))
                        scenes = [h.doc_id.removeprefix("scene:") for h in hits if h.kind == "scene"]
                    else:
                        rr = hybrid_search(state, db, q["query"], allowed, kinds=kinds, item_k=10, evidence_per_item=3)
                        items = [it.item_key for it in rr.items]
                        scenes = [d.doc_id.removeprefix("scene:") for it in rr.items for d in it.docs if d.kind == "scene"]
                    rel = set(q["relevant_item_keys"])
                    h1.append(hits_at_k(items, rel, 1))
                    h5.append(hits_at_k(items, rel, 5))
                    mr.append(mrr_at_k(items, rel, 10))
                    if q.get("expected_scene_ids") and kinds and "scene" in kinds:
                        s1.append(float(bool(scenes) and scenes[0] in q["expected_scene_ids"]))
                        s3.append(float(bool(set(scenes[:3]) & set(q["expected_scene_ids"]))))
            finally:
                state.dense = saved
            row = {"corpus": corpus, "retrieval": variant, "queries": len(queries), "hits@1": float(np.mean(h1)),
                   "hits@5": float(np.mean(h5)), "mrr@10": float(np.mean(mr))}
            if s1:
                row.update({"scene@1": float(np.mean(s1)), "scene@3": float(np.mean(s3))})
            out.append(row)
    return out


def run(settings, tag=None, **_) -> dict:
    from app.db.models import User
    from app.db.session import session_scope
    from app.services import chat as chat_svc
    from app.services.serving import load_state

    run = Run(settings, "chat", {"tag": tag})
    rng = np.random.default_rng(17)
    metrics: dict = {"sets": {}}
    rows = []
    with sandbox_settings(settings) as sb:
        state = load_state(sb)
        snap = state.snap
        uid = make_user(sb, "eval_chat_user")
        sets: dict[str, list[dict]] = {}
        lab = _labelled(settings)
        if lab:
            sets.update(_split_by_category(*lab))
        if settings.mode == "real":
            sets.update(_auto_sets(state, snap, rng))
        for name, queries in sets.items():
            per = []
            for q in queries:
                with session_scope(sb.db_url) as db:
                    user = db.get(User, uid)
                    sess = chat_svc.create_session(db, user, None)
                    t0 = time.perf_counter()
                    try:
                        out = chat_svc.post_message(state, db, user, sess.id, q["query"])
                    except Exception as exc:  # noqa: BLE001 - recorded as a failed task
                        run.failure({"set": name, "query_id": q["query_id"], "error": f"{type(exc).__name__}: {exc}"})
                        per.append({"tsr": 0.0})
                        continue
                    ms = (time.perf_counter() - t0) * 1000
                keys = [it["item_key"] for it in out["items"]]
                rel = set(q.get("relevant_item_keys") or [])
                hc = dict(q.get("hard_constraints") or {})
                compliant = all(_compliant(snap, k, hc) for k in keys) if keys else True
                fabricated_ep = _episode_claim(out["answer"], out["evidence"], snap)
                m = {"latency_ms": ms, "compliant": float(compliant), "fabricated_episode": float(fabricated_ep),
                     "n_items": len(keys)}
                if q.get("no_scene_claim"):
                    m["false_scene_claim"] = float(bool(SCENE_CLAIM.search(out["answer"])))
                if q.get("intent"):
                    m["intent_ok"] = float(out["intent"] == q["intent"])
                if rel and q.get("answerable") is not False:
                    m["hits@5"] = hits_at_k(keys, rel, 5)
                    m["mrr@10"] = mrr_at_k(keys, rel, 10)
                if q.get("expected_scene_ids"):
                    cited = _scene_ids(out["evidence"])
                    m["scene@3"] = float(bool(set(cited[:3]) & set(q["expected_scene_ids"])))
                    m["scene@1"] = float(bool(cited) and cited[0] in q["expected_scene_ids"])
                m["tsr"] = _task_success(q, out, keys, rel, compliant, fabricated_ep, m, snap)
                per.append(m)
                run.case({"set": name, "query_id": q["query_id"], "query": q["query"], "intent": out["intent"],
                          "expected_intent": q.get("intent"), "items": keys[:5], "answer_mode": out["answer_mode"],
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
            agg = {}
            for k in ("tsr", "hits@5", "mrr@10", "scene@1", "scene@3", "false_scene_claim", "compliant",
                      "fabricated_episode", "intent_ok"):
                vals = [p[k] for p in per if k in p]
                if vals:
                    agg[k] = bootstrap_ci(vals)
            lat = [p["latency_ms"] for p in per if "latency_ms" in p]
            agg["latency_ms_p50"] = float(np.percentile(lat, 50)) if lat else None
            agg["latency_ms_p95"] = float(np.percentile(lat, 95)) if lat else None
            metrics["sets"][name] = {"queries": len(queries), **agg}
            rows.append({"set": name, "queries": len(queries), **{k: (agg[k]["mean"] if isinstance(agg.get(k), dict)
                                                                       else agg.get(k))
                                                                   for k in ("tsr", "hits@5", "mrr@10", "scene@3",
                                                                             "false_scene_claim", "compliant",
                                                                             "fabricated_episode", "intent_ok",
                                                                             "latency_ms_p95")}})
        # ---- retrieval ablations on labelled queries (raw query text, content policy applied)
        if lab:
            allowed = chat_svc._policy_keys(state)
            desc = [q for q in lab[1] if q.get("intent") == "preference_recommendation" and q.get("relevant_item_keys")
                    and not q.get("hard_constraints")]
            scene_q = [q for q in lab[1] if q.get("intent") == "scene_lookup" and q.get("relevant_item_keys")
                       and q.get("answerable") is not False]
            with session_scope(sb.db_url) as db:
                if desc:
                    metrics["retrieval_ablation"] = _ablation(state, db, desc, allowed, [("all", None)])
                if scene_q:
                    corpora = [("scene", ["scene"]), ("synopsis", ["synopsis"]), ("scene+synopsis", ["scene", "synopsis"])]
                    metrics["scene_retrieval_ablation"] = _ablation(state, db, scene_q, allowed, corpora)
        llm_cap = state.cap("llm")
        metrics["plain_llm_vs_rag"] = {"status": "blocked_by_credentials" if llm_cap.status != "ready" else "not_run",
                                       "detail": llm_cap.detail}
    run.comparison(rows)
    lines = [f"# Chat-to-find (algorithm 4) - {settings.mode} data", "",
             "All answers here were produced in the configured answer mode (local evidence templates unless an LLM is "
             "configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.", ""]
    if settings.mode == "real":
        if lab and lab[0].startswith("team_package_"):
            lines += [f"> The `{lab[0]}:*` sets are the team data package's evaluation queries. Scene queries are "
                      "anchored to one scene record each (checked by an AI agent against cited sources, not by a "
                      "person); semantic and constraint queries count every catalogue title that satisfies the stated "
                      "filters as relevant, so Hits@5 there mostly measures constraint compliance. They are not "
                      "independent human relevance judgements.", ""]
        elif not lab:
            lines += ["> No labelled RAG query set was provided (data/evaluation/rag_queries.jsonl).", ""]
        lines += ["> The `auto_*` sets are generated from catalogue facts and test robustness, constraint compliance "
                  "and refusal behaviour; they are not relevance judgements.", ""]
    else:
        lines += ["> Synthetic generator queries validate the pipeline only.", ""]
    lines += [md_table(rows, ["set", "queries", "tsr", "hits@5", "mrr@10", "scene@3", "false_scene_claim", "compliant",
                              "fabricated_episode", "intent_ok", "latency_ms_p95"]), ""]
    if metrics.get("retrieval_ablation"):
        lines += ["## Retrieval ablation (descriptive labelled queries, no constraints)", "",
                  md_table(metrics["retrieval_ablation"], ["retrieval", "queries", "hits@5", "mrr@10"]), ""]
    if metrics.get("scene_retrieval_ablation"):
        lines += ["## Scene retrieval ablation (labelled scene queries, raw query text)", "",
                  "Item-level hits over all served titles; scene@k counts the expected scene record itself. The "
                  "synopsis-only rows show what plot-level matching finds without the scene layer.", "",
                  md_table(metrics["scene_retrieval_ablation"],
                           ["corpus", "retrieval", "queries", "hits@1", "hits@5", "mrr@10", "scene@1", "scene@3"]), ""]
    lines += [f"## Plain LLM vs RAG: {metrics['plain_llm_vs_rag']['status']}", "", metrics["plain_llm_vs_rag"]["detail"], ""]
    return run.finish(metrics, "\n".join(lines) + "\n", {"sets": {k: v["queries"] for k, v in metrics["sets"].items()}})
