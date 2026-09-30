"""Similar-titles experiment (algorithm 1): A weighted / B logistic / C BGE attributes / D hybrid.

Label sources are reported separately and never mixed:
  human_annotation (data/evaluation/similarity_labels.jsonl), mal_user_recommendations_weak (Jikan),
  synthetic_generator (synthetic mode) and franchise_sanity (confirmed sequel/prequel pairs: a sanity
  check, not general similarity truth). Co-high-rating pairs are never used.
All scorers rank the same candidate pool (served, recommendable, excluding the query) from the same
attribute inputs. `semantic_synopsis` uses synopsis text, i.e. different information, and is reported
as a separate variant. The replacement rule (Hits@5 +2pp and NDCG@5 not lower vs A) is applied on the
validation query group; test numbers are reported only.
"""

from __future__ import annotations

import json
import time

import numpy as np

from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci, hits_at_k, ndcg_at_k, recall_at_k
from evaluation.protocols import load_context
from pipelines.common import read_jsonl
from pipelines.fit import similarity_label_sources
from pipelines.processed_io import ProcessedDataset

K = 5


def _scorers(ctx):
    from app.recommendation.similarity.embedding import EmbeddingSimilarityScorer, HybridSimilarityScorer

    out = {"A_weighted_attributes": ctx.weighted}
    comps = ctx.manifest.get("components", {})
    if comps.get("logistic", {}).get("status") == "trained":
        import joblib

        from app.recommendation.similarity.logistic import LogisticSimilarityScorer

        out["B_logistic_similarity"] = LogisticSimilarityScorer(
            joblib.load(ctx.art / "models" / "logistic_similarity.joblib"), ctx.weighted, comps["logistic"])
    emb = comps.get("embeddings", {})
    if emb.get("status") == "trained":
        es = EmbeddingSimilarityScorer(np.load(ctx.art / "embeddings" / "item_attr.npy"),
                                       np.load(ctx.art / "embeddings" / "item_attr_mask.npy"), ctx.weighted, emb["info"])
        out["C_bge_attribute_embedding"] = es
        out["D_tag_semantic_hybrid"] = HybridSimilarityScorer(es, ctx.weighted, 0.6)
    syn = _synopsis_scorer(ctx)
    if syn is not None:
        out["variant_semantic_synopsis"] = syn
    return out


def _synopsis_scorer(ctx):
    """Mean of each title's synopsis-chunk vectors (different information than A-D)."""
    import pandas as pd

    from app.recommendation.similarity.embedding import EmbeddingSimilarityScorer

    vec_path = ctx.art / "index" / "doc_vectors.npy"
    if not vec_path.exists():
        return None
    docs = pd.read_parquet(ctx.art / "index" / "documents.parquet", columns=["item_key", "kind"])
    vecs = np.load(vec_path)
    sel = docs.kind.to_numpy() == "synopsis"
    full = np.zeros((ctx.snap.n, vecs.shape[1]), dtype=np.float32)
    counts = np.zeros(ctx.snap.n)
    for key, v in zip(docs.item_key.to_numpy()[sel], vecs[sel], strict=True):
        i = ctx.snap.index.get(key)
        if i is not None:
            full[i] += v
            counts[i] += 1
    mask = counts > 0
    full[mask] /= np.linalg.norm(full[mask], axis=1, keepdims=True)
    info = json.loads((ctx.art / "index" / "dense_info.json").read_text())
    s = EmbeddingSimilarityScorer(full, mask, ctx.weighted, {**info, "provider": info.get("provider")})
    s.scorer_id = f"semantic_synopsis_{info.get('provider')}"
    return s


def _franchise_labels(ctx) -> list[dict]:
    ds = ProcessedDataset.for_version(ctx.settings, ctx.version)
    rel = ds.table("relations")
    if not len(rel):
        return []
    rel = rel[(rel.to_entity_type == "anime") & rel.relation_type.isin(["sequel", "prequel"])]
    groups: dict[str, set[str]] = {}
    for a, b in zip(rel.from_item, rel.to_item, strict=True):
        groups.setdefault(a, set()).add(b)
    return [{"query_item": q, "positives": [{"item_key": p, "grade": 1} for p in sorted(ps)],
             "label_source": "franchise_sanity"} for q, ps in sorted(groups.items())]


def run(settings, tag=None, **_) -> dict:
    ctx = load_context(settings, load_models=False)
    run = Run(settings, "similar", {"k": K, "tag": tag})
    ds = ProcessedDataset.for_version(settings, ctx.version)
    sources = [(name, list(read_jsonl(path))) for name, path in similarity_label_sources(settings, ds)]
    fr = _franchise_labels(ctx)
    if fr:
        sources.append(("franchise_sanity", fr))
    groups_path = ctx.art / "models" / "similarity_query_groups.json"
    groups = json.loads(groups_path.read_text()) if groups_path.exists() else {}
    scorers = _scorers(ctx)
    pool = ctx.pool_mask
    metrics = {"sources": {}, "scorers": sorted(scorers)}
    rows = []
    decisions = []
    if not sources:
        run.status = "not_run"
        run.notes.append("No similarity labels are available (human annotation, weak MAL recommendations or "
                         "franchise relations).")
    for src, labels in sources:
        trained_on = ctx.manifest.get("components", {}).get("logistic", {}).get("label_source")
        per_split = {}
        for row in labels:
            q = ctx.snap.index.get(row["query_item"])
            if q is None or not ctx.snap.in_catalog[q]:
                run.failure({"source": src, "query": row["query_item"], "reason": "query not in served catalogue"})
                continue
            grades = {ctx.snap.index[p["item_key"]]: p.get("grade", 1) for p in row["positives"]
                      if p["item_key"] in ctx.snap.index and pool[ctx.snap.index[p["item_key"]]] and
                      p["item_key"] != row["query_item"]}
            if not grades:
                run.failure({"source": src, "query": row["query_item"], "reason": "no positive in candidate pool"})
                continue
            split = groups.get(row["query_item"], "test") if src == trained_on else "all"
            cand = np.flatnonzero(pool)
            cand = cand[cand != q]
            for name, scorer in scorers.items():
                t0 = time.perf_counter()
                res = scorer.score(ctx.snap, q, cand)
                s = np.nan_to_num(res.score, nan=-np.inf)
                top = cand[np.lexsort((ctx.snap.keys[cand], -res.coverage, -s))[:K]]
                top = top[np.isfinite(s[np.searchsorted(cand, top)])] if len(top) else top
                ms = (time.perf_counter() - t0) * 1000
                rel = set(grades)
                gains = [grades.get(int(t), 0) for t in top]
                m = {"hits@5": hits_at_k(top.tolist(), rel, K), "recall@5": recall_at_k(top.tolist(), rel, K),
                     "ndcg@5": ndcg_at_k(gains, sorted(grades.values(), reverse=True), K),
                     "coverage": float(np.mean(res.coverage[np.searchsorted(cand, top)])) if len(top) else 0.0,
                     "latency_ms": ms}
                per_split.setdefault(split, {}).setdefault(name, []).append(m)
                per_split[split].setdefault(f"_{name}_lists", set()).update(top.tolist())
                run.case({"source": src, "split": split, "query": row["query_item"], "scorer": name,
                          "top": [str(ctx.snap.keys[t]) for t in top], **{k: round(v, 5) for k, v in m.items()}})
        metrics["sources"][src] = {}
        for split, by in per_split.items():
            out = {}
            for name in scorers:
                vals = by.get(name, [])
                if not vals:
                    continue
                out[name] = {k: bootstrap_ci([v[k] for v in vals]) for k in ("hits@5", "recall@5", "ndcg@5", "coverage")}
                out[name]["latency_ms_p50"] = float(np.percentile([v["latency_ms"] for v in vals], 50))
                out[name]["catalog_coverage"] = round(len(by.get(f"_{name}_lists", set())) / max(1, int(pool.sum())), 5)
                rows.append({"label_source": src, "split": split, "scorer": name,
                             **{k: out[name][k]["mean"] for k in ("hits@5", "recall@5", "ndcg@5", "coverage")},
                             "queries": out[name]["hits@5"]["n"]})
            metrics["sources"][src][split] = out
            if split == "val":
                base = out.get("A_weighted_attributes")
                min_q = int(settings.recommendation.get("similar", {}).get("replacement_min_val_queries", 20))
                for name, o in out.items():
                    if name.startswith("A_") or base is None:
                        continue
                    n_q = int(o["hits@5"]["n"])
                    ok = (o["hits@5"]["mean"] - base["hits@5"]["mean"] >= 0.02 and
                          o["ndcg@5"]["mean"] >= base["ndcg@5"]["mean"])
                    # too few validation queries -> no decision (a single query can flip Hits@5 by 1.0)
                    decision = bool(ok) if n_q >= min_q else f"no decision (n={n_q} < {min_q})"
                    decisions.append({"source": src, "candidate": name, "replace_default": decision,
                                      "val_queries": n_q,
                                      "delta_hits@5": round(o["hits@5"]["mean"] - base["hits@5"]["mean"], 4),
                                      "delta_ndcg@5": round(o["ndcg@5"]["mean"] - base["ndcg@5"]["mean"], 4)})
    metrics["replacement_decisions_on_validation"] = decisions
    run.comparison(rows)
    lines = [f"# Similar titles (algorithm 1) - {settings.mode} data", "",
             f"Dataset `{ctx.version}`. Candidate pool: {int(pool.sum())} served, recommendable titles. K={K}. "
             "Each label source is evaluated separately. `split=val/test` applies to the source the logistic model was "
             "trained on (query groups kept franchise-disjoint); other sources are evaluated on all queries (`all`).", ""]
    if settings.mode == "synthetic":
        lines += ["> Synthetic generator labels validate the pipeline only; they partly reuse the tag structure the "
                  "attribute scorers see, so they are not an unbiased quality estimate.", ""]
    if rows:
        lines += [md_table(rows, ["label_source", "split", "scorer", "hits@5", "recall@5", "ndcg@5", "coverage", "queries"]), ""]
    if decisions:
        lines += ["## Replacement rule on validation (Hits@5 +2pp and NDCG@5 not lower than A)", "",
                  md_table(decisions, ["source", "candidate", "val_queries", "delta_hits@5", "delta_ndcg@5",
                                       "replace_default"]), "",
                  "The served default changes only by configuration; this table records what the rule says, and "
                  "it makes no decision below the minimum number of validation queries.", ""]
    lines += ["`coverage` is the mean metadata coverage (share of weighted attribute dimensions that were comparable) "
              "of the returned titles; catalog coverage per scorer is in metrics.json.", ""]
    if "franchise_sanity" in metrics.get("sources", {}):
        lines += ["`franchise_sanity` asks whether a title's confirmed sequel/prequel appears in its top 5. It is a sanity "
                  "check of franchise awareness, not a general similarity benchmark: the synopsis variant does well "
                  "here because franchise entries share character names.", ""]
    lines += [f"- note: {n}" for n in run.notes]
    return run.finish(metrics, "\n".join(lines) + "\n", {"label_sources": [s for s, _ in sources]})
