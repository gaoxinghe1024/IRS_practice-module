"""Chat-to-find evaluation (algorithm 4).

Query sets (each reported separately):
  labelled         data/evaluation/rag_queries.jsonl (human) or the synthetic generator's queries
  auto_titles      REAL mode, auto-generated: exact / alias / one-typo title lookups (ground truth by construction)
  auto_constraints REAL mode, auto-generated: "a <genre> anime with at most N episodes, no <genre2>"
  auto_relations   REAL mode, auto-generated from confirmed sequel relations
  auto_no_episode  REAL mode: "which episode of <title> has the ... scene" (no scene data exists: an
                   episode number in the answer is a fabrication)
Auto-generated sets are NOT human-annotated relevance judgements; they test robustness, constraint
compliance and refusal behaviour.

Retrieval ablation (labelled descriptive queries only): lexical, dense and hybrid (RRF) item-level
Hits@5 / MRR@10 over all served titles.
Plain LLM vs RAG needs a configured model; without credentials it is reported as blocked, never faked.
Task success (TSR): answerable -> at least one relevant title in the top 5 and every returned title
satisfies the hard constraints; unanswerable -> no fabricated episode / no results claimed; contradictory
-> a clarification is asked.
"""

from __future__ import annotations

import re
import time

import numpy as np

from app.retrieval.evidence import hybrid_search
from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci, hits_at_k, mrr_at_k
from evaluation.sandbox import make_user, sandbox_settings
from pipelines.common import read_jsonl

EPISODE_CLAIM = re.compile(r"\bepisode\s*(?:#|no\.?\s*)?\d+\b|第\s*\d+\s*集", re.I)


def _compliant(snap, key: str, hc: dict) -> bool:
    i = snap.index[key]
    genres = set(snap.tags["genre"].tags_of(i)) if "genre" in snap.tags else set()
    ep = snap.episodes[i]
    if hc.get("max_episodes") is not None and not (np.isfinite(ep) and ep <= hc["max_episodes"]):
        return False
    if hc.get("min_episodes") is not None and not (np.isfinite(ep) and ep >= hc["min_episodes"]):
        return False
    if any(g not in genres for g in hc.get("include_genres", [])):
        return False
    if any(g in genres for g in hc.get("exclude_genres", [])):
        return False
    return True


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
    seen = set()
    for a, b in rows:
        if a in seen or a not in snap.index:
            continue
        seen.add(a)
        rels.append({"query_id": f"rel-{a}", "query": f"what is the sequel to {snap.titles[snap.index[a]]}?",
                     "intent": "relation_query", "relevant_item_keys": [b], "hard_constraints": {}, "answerable": True})
    noep = [{"query_id": f"noep-{i}", "query": f"which episode of {snap.titles[i]} has the scene where they fight "
             "on the rooftop in the rain?", "intent": "scene_lookup", "relevant_item_keys": [], "hard_constraints": {},
             "answerable": False} for i in picks[:30]]
    return {"auto_titles": titles, "auto_constraints": cons, "auto_relations": rels[:60], "auto_no_episode": noep}


def _labelled(settings) -> tuple[str, list[dict]] | None:
    human = settings.data_root / "evaluation" / "rag_queries.jsonl"
    if human.exists():
        return "labelled_human", list(read_jsonl(human))
    if settings.mode == "synthetic":
        p = settings.synthetic_dir / "evaluation" / "rag_queries.jsonl"
        if p.exists():
            return "labelled_synthetic_generator", list(read_jsonl(p))
    return None


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
            sets[lab[0]] = lab[1]
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
                fabricated_ep = bool(EPISODE_CLAIM.search(out["answer"])) and not any(
                    e["source_type"] == "scene_verified" for e in out["evidence"])
                m = {"latency_ms": ms, "compliant": float(compliant), "fabricated_episode": float(fabricated_ep),
                     "intent_ok": float(out["intent"] == q.get("intent")), "n_items": len(keys)}
                if q.get("answerable") is False:
                    if q.get("hard_constraints", {}).get("min_episodes") is not None and "max_episodes" in q.get(
                            "hard_constraints", {}):
                        ok = bool(out.get("clarification"))
                    else:
                        ok = not fabricated_ep
                    m["tsr"] = float(ok)
                elif rel:
                    m["hits@5"] = hits_at_k(keys, rel, 5)
                    m["mrr@10"] = mrr_at_k(keys, rel, 10)
                    m["tsr"] = float(m["hits@5"] == 1.0 and compliant and not fabricated_ep)
                else:
                    m["tsr"] = float(bool(keys) and compliant and not fabricated_ep)
                per.append(m)
                run.case({"set": name, "query_id": q["query_id"], "query": q["query"], "intent": out["intent"],
                          "expected_intent": q.get("intent"), "items": keys[:5], "answer_mode": out["answer_mode"],
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
            agg = {}
            for k in ("tsr", "hits@5", "mrr@10", "compliant", "fabricated_episode", "intent_ok"):
                vals = [p[k] for p in per if k in p]
                if vals:
                    agg[k] = bootstrap_ci(vals)
            lat = [p["latency_ms"] for p in per if "latency_ms" in p]
            agg["latency_ms_p50"] = float(np.percentile(lat, 50)) if lat else None
            agg["latency_ms_p95"] = float(np.percentile(lat, 95)) if lat else None
            metrics["sets"][name] = {"queries": len(queries), **agg}
            rows.append({"set": name, "queries": len(queries), **{k: (agg[k]["mean"] if isinstance(agg.get(k), dict)
                                                                       else agg.get(k))
                                                                   for k in ("tsr", "hits@5", "mrr@10", "compliant",
                                                                             "fabricated_episode", "intent_ok",
                                                                             "latency_ms_p95")}})
        # ---- retrieval ablation on descriptive labelled queries
        abl_rows = []
        if lab:
            desc = [q for q in lab[1] if q.get("intent") == "preference_recommendation" and q.get("relevant_item_keys")
                    and not q.get("hard_constraints")]
            with session_scope(sb.db_url) as db:
                for variant in ("lexical", "dense", "hybrid"):
                    hs, mr = [], []
                    saved_dense = state.dense
                    if variant == "lexical":
                        state.dense = None
                    for q in desc:
                        if variant == "dense" and saved_dense is not None:
                            hits = saved_dense.search(q["query"], None, top_n=50)
                            items = list(dict.fromkeys(h.item_key for h in hits))
                        else:
                            items = [it.item_key for it in hybrid_search(state, db, q["query"], None, item_k=10).items]
                        hs.append(hits_at_k(items, set(q["relevant_item_keys"]), 5))
                        mr.append(mrr_at_k(items, set(q["relevant_item_keys"]), 10))
                    state.dense = saved_dense
                    if variant == "dense" and state.dense is None:
                        continue
                    abl_rows.append({"retrieval": variant, "queries": len(desc), "hits@5": float(np.mean(hs)) if hs else None,
                                     "mrr@10": float(np.mean(mr)) if mr else None})
            metrics["retrieval_ablation"] = abl_rows
        llm_cap = state.cap("llm")
        metrics["plain_llm_vs_rag"] = {"status": "blocked_by_credentials" if llm_cap.status != "ready" else "not_run",
                                       "detail": llm_cap.detail}
    run.comparison(rows)
    lines = [f"# Chat-to-find (algorithm 4) - {settings.mode} data", "",
             "All answers here were produced in the configured answer mode (local evidence templates unless an LLM is "
             "configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.", ""]
    if settings.mode == "real":
        lines += ["> No human-annotated RAG query set was provided (data/evaluation/rag_queries.jsonl). The `auto_*` "
                  "sets are generated from catalogue facts and test robustness, constraint compliance and refusal "
                  "behaviour; they are not relevance judgements.", ""]
    else:
        lines += ["> Synthetic generator queries validate the pipeline only.", ""]
    lines += [md_table(rows, ["set", "queries", "tsr", "hits@5", "mrr@10", "compliant", "fabricated_episode", "intent_ok",
                              "latency_ms_p95"]), ""]
    if metrics.get("retrieval_ablation"):
        lines += ["## Retrieval ablation (descriptive labelled queries, no constraints)", "",
                  md_table(metrics["retrieval_ablation"], ["retrieval", "queries", "hits@5", "mrr@10"]), ""]
    lines += [f"## Plain LLM vs RAG: {metrics['plain_llm_vs_rag']['status']}", "", metrics["plain_llm_vs_rag"]["detail"], ""]
    return run.finish(metrics, "\n".join(lines) + "\n", {"sets": {k: v["queries"] for k, v in metrics["sets"].items()}})
