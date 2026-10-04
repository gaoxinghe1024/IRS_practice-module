"""Offline training of every served / compared model from the ranking-protocol train split.

artifacts/<mode>/<dataset_version>/
    manifest.json                      versions, split fingerprint, params, metrics, checksums
    stats/train_interactions.npy       per catalogue index (train split only)
    models/cf_items.npy                catalogue index of each CF item (train items)
    models/icf.npz                     item neighbours (exact adjusted cosine)
    models/mf.npz                      biased MF
    models/ncf.pt                      NCF explicit (ml extra)
    models/logistic_similarity.joblib  (when similarity labels exist)
    models/gbdt_collection.joblib      (when collection events exist)
    models/two_tower.pt                (when collection events exist; deployment gated)
    embeddings/item_attr.npy (+ mask)  attribute-text embeddings for algorithm 1 C/D

Nothing is trained on validation or test rows. MF/NCF early stopping uses a
seeded 2% slice carved out of the train split itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.recommendation.features import snapshot_from_processed
from pipelines.common import (
    dependency_versions,
    latest_dataset_version,
    read_jsonl,
    rel,
    sha256_file,
    utc_now_iso,
    write_json,
)
from pipelines.processed_io import ProcessedDataset, mix_hash
from pipelines.split import load_split, split_manifest

ALL_COMPONENTS = ("icf", "mf", "ncf", "embeddings", "logistic", "gbdt", "two_tower")


def artifact_dir(settings: Settings, version: str) -> Path:
    return settings.artifacts_dir / version


def _device(settings: Settings) -> str:
    dev = settings.retrieval.get("embedding", {}).get("device", "auto")
    if dev != "auto":
        return dev
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def similarity_label_sources(settings: Settings, ds: ProcessedDataset) -> list[tuple[str, Path]]:
    """Label files in priority order: human annotation, then declared weak / generator sources."""
    out = []
    human = settings.data_root / "evaluation" / "similarity_labels.jsonl"
    if settings.mode == "real" and human.exists():
        out.append(("human_annotation", human))
    if settings.mode == "synthetic":
        p = settings.synthetic_dir / "evaluation" / "similarity_labels.jsonl"
        if p.exists():
            out.append(("synthetic_generator", p))
    # weak MAL-recommendation labels: the algorithm-1 benchmark's ground truth (import-similarity-labels) when
    # present, else the few pairs derived from the Jikan cache during preprocessing
    mal = settings.data_root / "evaluation" / "similarity_labels_mal.jsonl"
    weak = ds.labels_path("similarity_mal_recommendations")
    if settings.mode == "real" and mal.exists() and mal.stat().st_size > 0:
        out.append(("mal_user_recommendations_weak", mal))
    elif weak.exists() and weak.stat().st_size > 0:
        out.append(("mal_user_recommendations_weak", weak))
    return out


def collection_event_source(settings: Settings) -> tuple[str, Path] | None:
    if settings.mode == "synthetic":
        p = settings.synthetic_dir / "evaluation" / "collection_events.jsonl"
        return ("synthetic_generator", p) if p.exists() else None
    p = settings.data_root / "evaluation" / "collection_events.jsonl"
    return ("real_collection_events", p) if p.exists() else None


def franchise_components(ds: ProcessedDataset) -> dict[str, str]:
    """Union-find over confirmed anime-anime relations (keeps a franchise in one split)."""
    rel_df = ds.table("relations")
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if len(rel_df):
        for a, b, t in rel_df[["from_item", "to_item", "to_entity_type"]].itertuples(index=False):
            if t == "anime" and a and b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {k: find(k) for k in parent}


def train(settings: Settings, version: str | None = None, components: tuple[str, ...] = ALL_COMPONENTS,
          log=print) -> dict[str, Any]:
    t_all = time.time()
    version = version or latest_dataset_version(settings)
    ds = ProcessedDataset.for_version(settings, version)
    snap = snapshot_from_processed(ds)
    out = artifact_dir(settings, version)
    for sub in ("models", "embeddings", "stats", "catalog"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # self-contained catalogue copy: serving indices must equal training indices
    import shutil

    for name in ("catalog", "item_tags", "aliases"):
        shutil.copyfile(ds.root / f"{name}.parquet", out / "catalog" / f"{name}.parquet")
    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    split_m = split_manifest(settings, "ranking", version)
    manifest.update({"dataset_mode": settings.mode, "dataset_version": version,
                     "split": {"protocol": "ranking", "fingerprint": split_m["fingerprint"]},
                     "dependencies": dependency_versions(), "updated_at": utc_now_iso()})
    comps = manifest.setdefault("components", {})
    rec = settings.recommendation
    seed = int(settings.pipeline.get("split_seed", 20260930))

    tr = load_split(settings, "ranking", "train", version)
    counts = np.bincount(tr.item_idx.to_numpy(), minlength=snap.n).astype(np.int64)
    np.save(out / "stats" / "train_interactions.npy", counts)
    snap.with_train_interactions(counts)
    cf_items = np.unique(tr.item_idx.to_numpy())
    cf_items = cf_items[snap.in_catalog[cf_items]]
    np.save(out / "models" / "cf_items.npy", cf_items)
    cat_to_cf = np.full(snap.n, -1, dtype=np.int64)
    cat_to_cf[cf_items] = np.arange(len(cf_items))
    tr = tr[cat_to_cf[tr.item_idx.to_numpy()] >= 0]
    users, u_idx = np.unique(tr.offline_user_id.to_numpy(), return_inverse=True)
    i_idx = cat_to_cf[tr.item_idx.to_numpy()]
    r = tr.rating.to_numpy().astype(np.float32)
    np.save(out / "models" / "offline_users.npy", users)
    manifest["cf"] = {"n_items": int(len(cf_items)), "n_users": int(len(users)), "n_ratings": int(len(r)),
                      "dense_float32_gb_if_materialised": round(len(users) * len(cf_items) * 4 / 1e9, 2)}
    log(f"[train] {version}: cf_items={len(cf_items)} users={len(users)} ratings={len(r):,}")

    # seeded early-stopping slice carved from train (never val/test)
    es = mix_hash(u_idx.astype(np.int64) * 7_919 + i_idx, seed + 1) < 0.02
    fit_rows, es_rows = ~es, es

    if "icf" in components:
        from app.recommendation.recall.icf import fit_icf

        p = rec.get("icf", {})
        t0 = time.time()
        model = fit_icf(u_idx, i_idx, r, len(users), len(cf_items), min_co_rated=int(p.get("min_co_rated", 3)),
                        shrinkage=float(p.get("shrinkage", 10)), top_n=int(p.get("neighbors", 150)),
                        block_size=int(p.get("block_size", 1024)), log=log)
        model.save(out / "models" / "icf.npz")
        comps["icf"] = {"status": "trained", "engine_id": "icf_adjusted_cosine", "params": model.params,
                        "nnz": model.nnz, "seconds": round(time.time() - t0, 1),
                        "avg_neighbours": round(model.nnz / max(1, len(cf_items)), 1),
                        "file": "models/icf.npz", "sha256": sha256_file(out / "models" / "icf.npz")}

    if "mf" in components:
        from app.recommendation.recall.matrix_factorization import fit_mf

        p = dict(rec.get("mf", {}))
        t0 = time.time()
        model = fit_mf(u_idx[fit_rows], i_idx[fit_rows], r[fit_rows], len(users), len(cf_items),
                       val=(u_idx[es_rows], i_idx[es_rows], r[es_rows]), seed=seed, log=log, **p)
        model.save(out / "models" / "mf.npz")
        comps["mf"] = {"status": "trained", "engine_id": "mf_biased_sgd", "params": model.params,
                       "history": model.history, "seconds": round(time.time() - t0, 1),
                       "early_stopping": "2% seeded slice of ranking-train",
                       "file": "models/mf.npz", "sha256": sha256_file(out / "models" / "mf.npz")}

    if "ncf" in components:
        from app.recommendation.recall.ncf import fit_ncf, torch_available

        if not torch_available():
            comps["ncf"] = {"status": "not_configured", "reason": "torch not installed (ml extra)"}
        else:
            p = dict(rec.get("ncf", {}))
            t0 = time.time()
            model = fit_ncf(u_idx[fit_rows], i_idx[fit_rows], r[fit_rows], len(users), len(cf_items),
                            val=(u_idx[es_rows], i_idx[es_rows], r[es_rows]), seed=seed, log=log, **p)
            model.save(out / "models" / "ncf.pt")
            comps["ncf"] = {"status": "trained", "engine_id": "ncf_explicit", "params": model.params,
                            "history": model.history, "seconds": round(time.time() - t0, 1),
                            "serving_note": "ID-only: application users are unsupported_new_user -> fallback",
                            "file": "models/ncf.pt", "sha256": sha256_file(out / "models" / "ncf.pt")}

    if "embeddings" in components:
        comps["embeddings"] = _train_embeddings(settings, snap, out, log)

    emb_scorer = None
    if (out / "embeddings" / "item_attr.npy").exists():
        from app.recommendation.similarity.embedding import EmbeddingSimilarityScorer
        from app.recommendation.similarity.weighted import WeightedAttributeScorer

        emb_scorer = EmbeddingSimilarityScorer(np.load(out / "embeddings" / "item_attr.npy"),
                                               np.load(out / "embeddings" / "item_attr_mask.npy"),
                                               WeightedAttributeScorer(), comps.get("embeddings", {}).get("info", {}))

    if "logistic" in components:
        comps["logistic"] = _train_logistic(settings, ds, snap, out, seed, log)

    if "gbdt" in components or "two_tower" in components:
        src = collection_event_source(settings)
        if src is None:
            for c in ("gbdt", "two_tower"):
                if c in components:
                    comps[c] = {"status": "limited_by_data",
                                "reason": "no collection events (data/evaluation/collection_events.jsonl)"}
        else:
            from app.recommendation.collection.cases import build_cases, split_owner

            events = list(read_jsonl(src[1]))
            cases = build_cases(snap, events, seed)
            parts = {c.case_id: split_owner(c.owner, seed) for c in cases}
            split_cases = {s: [c for c in cases if parts[c.case_id] == s] for s in ("train", "val", "test")}
            n_inter = sum(len(e["items"]) for e in events)
            if "gbdt" in components:
                comps["gbdt"] = _train_gbdt(settings, snap, split_cases, out, seed, src[0], log)
            if "two_tower" in components:
                comps["two_tower"] = _train_two_tower(settings, snap, split_cases, out, seed, src[0], n_inter, log)

    del emb_scorer
    manifest["files"] = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*"))
                         if p.is_file() and p.name != "manifest.json" and "index/" not in str(p.relative_to(out))}
    manifest["train_seconds"] = round(time.time() - t_all, 1)
    write_json(manifest_path, manifest)
    log(f"[train] done in {manifest['train_seconds']}s -> {rel(out)}")
    return manifest


def _train_embeddings(settings: Settings, snap, out: Path, log) -> dict:
    from app.recommendation.similarity.embedding import TEMPLATE_VERSION, attribute_text
    from app.retrieval.embeddings import BGEEncoder, choose_provider, sentence_transformers_available

    cfg = settings.retrieval.get("embedding", {})
    provider = choose_provider(cfg)
    if provider != "bge" or not sentence_transformers_available():
        return {"status": "not_configured", "reason": "BGE requires the ml extra (sentence-transformers); "
                "algorithm 1 C/D stay unavailable (no substitute is labelled as BGE)"}
    texts, rows = [], []
    for i in range(snap.n):
        if not snap.in_catalog[i]:
            continue
        t = attribute_text(snap, i)
        if t:
            texts.append(t)
            rows.append(i)
    t0 = time.time()
    try:
        enc = BGEEncoder(cfg.get("model", "BAAI/bge-small-en-v1.5"), batch_size=int(cfg.get("batch_size", 64)),
                         device=_device(settings))
    except Exception as exc:  # model download / load failure is reported, not hidden
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    vecs = enc.encode_documents(texts)
    full = np.zeros((snap.n, vecs.shape[1]), dtype=np.float32)
    mask = np.zeros(snap.n, dtype=bool)
    full[rows] = vecs
    mask[rows] = True
    np.save(out / "embeddings" / "item_attr.npy", full)
    np.save(out / "embeddings" / "item_attr_mask.npy", mask)
    log(f"[train] attribute embeddings: {len(rows)} items in {time.time() - t0:.1f}s ({enc.info()['model']})")
    return {"status": "trained", "info": enc.info(), "template": TEMPLATE_VERSION, "items": len(rows),
            "missing_policy": "missing attributes omitted from the text", "device": _device(settings),
            "seconds": round(time.time() - t0, 1)}


def _train_logistic(settings: Settings, ds, snap, out: Path, seed: int, log) -> dict:
    import joblib

    from app.recommendation.similarity.logistic import fit_logistic, group_queries, split_franchise_overlap
    from app.recommendation.similarity.weighted import WeightedAttributeScorer
    from evaluation.metrics import ndcg_at_k

    sources = similarity_label_sources(settings, ds)
    if not sources:
        return {"status": "limited_by_data", "reason": "no similarity labels (human or declared weak source)"}
    name, path = sources[0]
    rows = [r for r in read_jsonl(path) if r.get("query_item") in snap.index]
    components = franchise_components(ds)
    groups = group_queries(rows, components, seed)
    split_rows = {s: [r for r in rows if groups[r["query_item"]] == s] for s in ("train", "val", "test")}
    write_json(out / "models" / "similarity_query_groups.json", groups)
    counts = {s: len(v) for s, v in split_rows.items()}
    sim_cfg = settings.recommendation.get("similar", {})
    need_train = int(sim_cfg.get("logistic_min_train_queries", 20))
    need_val = int(sim_cfg.get("logistic_min_val_queries", 5))
    stale = out / "models" / "logistic_similarity.joblib"
    if counts["train"] < need_train or counts["val"] < need_val:
        if stale.exists():
            stale.unlink()                 # never leave a model from an earlier run next to a limited status
        log(f"[train] logistic similarity: limited_by_data ({name}: {counts}; need >= {need_train} train / "
            f"{need_val} validation query titles)")
        return {"scorer_id": "logistic_similarity", "status": "limited_by_data", "label_source": name,
                "label_file": rel(path), "queries": counts,
                "reason": f"only {counts['train']} training / {counts['val']} validation query titles with "
                          f"similarity labels from {name}; at least {need_train} / {need_val} are required"}
    base = WeightedAttributeScorer(settings.recommendation.get("similar", {}).get("weights"),
                                   settings.recommendation.get("similar", {}).get("tau_episodes", 12.0))
    pool = np.flatnonzero(snap.recommendable & snap.in_catalog & ~snap.explicit)

    def eval_fn(scorer, val_rows):
        vals = []
        for row in val_rows:
            q = snap.index[row["query_item"]]
            cand = pool[pool != q]
            res = scorer.score(snap, q, cand)
            s = np.nan_to_num(res.score, nan=-np.inf)
            top = cand[np.lexsort((snap.keys[cand], -s))[:5]]
            grades = {snap.index[p["item_key"]]: p.get("grade", 1) for p in row["positives"]
                      if p["item_key"] in snap.index}
            vals.append(ndcg_at_k([grades.get(int(t), 0) for t in top], sorted(grades.values(), reverse=True), 5))
        return float(np.mean(vals)) if vals else float("nan")

    model, meta = fit_logistic(snap, base, split_rows["train"], split_rows["val"], pool, seed=seed, eval_fn=eval_fn)
    meta.update({"label_source": name, "label_file": rel(path), "queries": counts,
                 "split_rule": "pre-registered split in the label file" if all(r.get("split") for r in rows)
                 else "franchise-grouped hash split",
                 "franchise_overlap": split_franchise_overlap(groups, components)})
    if model is not None:
        joblib.dump(model, out / "models" / "logistic_similarity.joblib")
        meta["file"] = "models/logistic_similarity.joblib"
        meta["sha256"] = sha256_file(out / "models" / "logistic_similarity.joblib")
        log(f"[train] logistic similarity ({name}): C={meta['C']} {meta['selection']}")
    return meta


def _train_gbdt(settings: Settings, snap, split_cases, out: Path, seed: int, source: str, log) -> dict:
    import joblib

    from app.recommendation.collection.cases import sample_negatives
    from app.recommendation.collection.gradient_boosting import collection_features, fit_gbdt

    cfg = settings.recommendation.get("gbdt", {})
    rng = np.random.default_rng(seed)
    k_neg = int(cfg.get("negatives_per_positive", 4))

    def rows(cases):
        X, y = [], []
        for c in cases:
            if not c.target_eligible:
                continue
            neg = sample_negatives(c, k_neg, rng)
            cand = np.concatenate([[c.target], neg])
            X.append(collection_features(snap, c.profile, cand))
            y.append(np.r_[1, np.zeros(len(neg))])
        return (np.vstack(X), np.concatenate(y)) if X else (np.zeros((0, 12)), np.zeros(0))

    Xtr, ytr = rows(split_cases["train"])
    Xva, yva = rows(split_cases["val"])
    if len(set(ytr.tolist())) < 2 or len(set(yva.tolist())) < 2:
        return {"status": "limited_by_data", "reason": "too few collection cases"}
    model, meta = fit_gbdt(Xtr, ytr, Xva, yva, int(cfg.get("max_iter", 200)), float(cfg.get("learning_rate", 0.05)),
                           seed)
    joblib.dump(model, out / "models" / "gbdt_collection.joblib")
    meta.update({"status": "trained", "label_source": source, "train_rows": int(len(ytr)),
                 "cases": {s: len(v) for s, v in split_cases.items()}, "file": "models/gbdt_collection.joblib",
                 "sha256": sha256_file(out / "models" / "gbdt_collection.joblib")})
    log(f"[train] gbdt collection scorer: val AUC(sampled)={meta['val_auc_sampled']}")
    return meta


def _train_two_tower(settings: Settings, snap, split_cases, out: Path, seed: int, source: str, n_inter: int,
                     log) -> dict:
    from app.recommendation.collection.two_tower import fit_two_tower
    from app.recommendation.recall.ncf import torch_available

    cfg = settings.recommendation.get("two_tower", {})
    if not torch_available():
        return {"status": "not_configured", "reason": "torch not installed (ml extra)"}
    gate = int(cfg.get("min_interactions", 10000))
    scorer = fit_two_tower(snap, split_cases["train"], split_cases["val"], embedding_dim=int(cfg.get("embedding_dim", 32)),
                           epochs=int(cfg.get("epochs", 15)), seed=seed, log=log)
    scorer.save(out / "models" / "two_tower.pt")
    meets_gate = n_inter >= gate
    return {"status": "trained", **scorer.meta, "label_source": source, "interactions": n_inter,
            "data_gate": gate, "meets_data_gate": meets_gate,
            "deployable": False, "deploy_note": "enabled only when the gate passes AND validation beats the "
            "default scorer (see evaluation); two_tower.enabled stays false by default",
            "file": "models/two_tower.pt", "sha256": sha256_file(out / "models" / "two_tower.pt")}
