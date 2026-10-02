"""The in-memory serving state for the active dataset version.

Everything a request needs from offline artifacts (catalogue snapshot, models,
vector indexes) is loaded together and verified against the activation record
(manifest checksum + per-file checksums). A request takes one reference to the
current state and uses it throughout, so a concurrent activation can never make
it mix a new index with an old ID mapping. When the database reports a newer
active version, a fresh state is built and swapped in as a whole.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from app.core.config import PROJECT_ROOT, Settings
from app.core.logging import get_logger, log_event
from app.db import models as M
from app.db.session import get_engine
from app.recommendation.constraints import ConstraintEngine, build_engine
from app.recommendation.features import CatalogSnapshot, build_snapshot
from app.retrieval.titles import TitleResolver

log = get_logger("serving")

STATUSES = ("ready", "degraded", "not_configured", "missing_data", "artifact_mismatch", "failed",
            "limited_by_data", "limited_by_credentials")


@dataclass
class Capability:
    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, **({"data": self.data} if self.data else {})}


@dataclass
class ServingState:
    settings: Settings
    dataset_mode: str
    dataset_version: str | None
    artifact_dir: Path | None
    manifest: dict
    snap: CatalogSnapshot | None = None
    constraints: ConstraintEngine | None = None
    titles: TitleResolver | None = None
    cf_items: np.ndarray | None = None
    cat_to_cf: np.ndarray | None = None
    icf: Any = None
    mf: Any = None
    ncf: Any = None
    similarity: dict[str, Any] = field(default_factory=dict)
    algorithm2: Any = None
    dense: Any = None
    images: Any = None
    capabilities: dict[str, Capability] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)

    def cap(self, name: str) -> Capability:
        return self.capabilities.get(name, Capability(name, "not_configured", "unknown capability"))

    def set_cap(self, name: str, status: str, detail: str, **data) -> None:
        assert status in STATUSES, status
        self.capabilities[name] = Capability(name, status, detail, data)

    @property
    def ready(self) -> bool:
        return self.snap is not None


def _verify(art: Path, manifest: dict, rel_path: str) -> bool:
    from pipelines.common import sha256_file

    expected = manifest.get("files", {}).get(rel_path) or manifest.get("index", {}).get("files", {}).get(rel_path)
    p = art / rel_path
    return p.exists() and expected is not None and sha256_file(p) == expected


def read_active(engine) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(select(M.SystemState.value).where(M.SystemState.key == "active_dataset")).first()
    return row[0] if row else None


def load_state(settings: Settings) -> ServingState:
    from pipelines.common import sha256_file

    engine = get_engine(settings.db_url)
    try:
        active = read_active(engine)
    except Exception as exc:  # schema missing
        active = None
        log_event(log, "no active dataset", error=type(exc).__name__)
    state = ServingState(settings, settings.mode, None, None, {})
    if not active:
        state.set_cap("catalog", "missing_data", "No dataset has been activated. Run `make seed` (synthetic) or "
                      "`make preprocess-real split-real train build-index activate-real` (real).")
        return state
    art = PROJECT_ROOT / active["artifact_dir"]
    state.dataset_version = active["dataset_version"]
    state.artifact_dir = art
    mpath = art / "manifest.json"
    if not mpath.exists() or sha256_file(mpath) != active.get("manifest_sha256"):
        state.set_cap("catalog", "artifact_mismatch", "The active artifact bundle does not match the activation "
                      "record. Re-run activation for this dataset version.")
        return state
    manifest = json.loads(mpath.read_text())
    state.manifest = manifest
    rec = settings.recommendation

    # ---------------------------------------------------------------- catalogue snapshot
    cat = pd.read_parquet(art / "catalog" / "catalog.parquet")
    tags = pd.read_parquet(art / "catalog" / "item_tags.parquet")
    aliases = pd.read_parquet(art / "catalog" / "aliases.parquet")
    snap = build_snapshot(cat, tags, state.dataset_version, settings.mode)
    counts_path = art / "stats" / "train_interactions.npy"
    snap.with_train_interactions(np.load(counts_path) if _verify(art, manifest, "stats/train_interactions.npy") else None)
    state.snap = snap
    served = set(snap.keys[snap.in_catalog & ~snap.explicit].tolist())
    state.constraints = build_engine(snap, aliases)
    pop = {k: float(m) for k, m in zip(snap.keys, np.nan_to_num(snap.members), strict=True)}
    state.titles = TitleResolver(list(aliases[["item_key", "alias", "kind"]].itertuples(index=False, name=None)),
                                 dict(zip(snap.keys, snap.titles, strict=True)), pop, served)
    state.set_cap("catalog", "ready", f"{len(served)} titles served ({int(snap.recommendable.sum())} recommendable)",
                  dataset_version=state.dataset_version)

    # ---------------------------------------------------------------- CF models
    if _verify(art, manifest, "models/cf_items.npy"):
        state.cf_items = np.load(art / "models" / "cf_items.npy")
        state.cat_to_cf = np.full(snap.n, -1, dtype=np.int64)
        state.cat_to_cf[state.cf_items] = np.arange(len(state.cf_items))
    if state.cf_items is not None and _verify(art, manifest, "models/icf.npz"):
        from app.recommendation.recall.icf import ICFModel, ICFScorer

        icf_cfg = rec.get("icf", {})
        state.icf = ICFScorer(ICFModel.load(art / "models" / "icf.npz"),
                              damping=float(icf_cfg.get("prediction_damping", 0.0)),
                              prior_weight=float(icf_cfg.get("user_mean_prior_weight", 0.0)))
        detail = f"{state.icf.model.nnz} item-neighbour edges"
        if state.icf.prior_weight > 0 and state.icf.model.params.get("global_train_mean") is None:
            detail += "; user-mean prior ignored (this ICF artifact has no global mean: retrain ICF)"
        state.set_cap("recall_icf", "ready", detail)
    else:
        state.set_cap("recall_icf", "missing_data", "ICF neighbours not trained for this version")
    if state.cf_items is not None and _verify(art, manifest, "models/mf.npz"):
        from app.recommendation.recall.matrix_factorization import MFModel

        state.mf = MFModel.load(art / "models" / "mf.npz")
        state.set_cap("recall_mf", "ready", "biased MF; application users are folded in by ridge regression")
    else:
        state.set_cap("recall_mf", "missing_data", "MF not trained for this version")
    ncf_meta = manifest.get("components", {}).get("ncf", {})
    if ncf_meta.get("status") == "trained":
        state.set_cap("recall_ncf", "degraded", "ID-only NCF: application users are unsupported_new_user; the home "
                      "feed falls back to ICF / content. Used in offline comparisons.")
    else:
        state.set_cap("recall_ncf", ncf_meta.get("status", "not_configured"), ncf_meta.get("reason", "not trained"))

    # ---------------------------------------------------------------- similarity scorers
    from app.recommendation.similarity.weighted import WeightedAttributeScorer

    sim_cfg = rec.get("similar", {})
    base = WeightedAttributeScorer(sim_cfg.get("weights"), sim_cfg.get("tau_episodes", 12.0))
    state.similarity["weighted_attributes"] = base
    lg = manifest.get("components", {}).get("logistic", {})
    if lg.get("status") == "trained" and _verify(art, manifest, "models/logistic_similarity.joblib"):
        import joblib

        from app.recommendation.similarity.logistic import LogisticSimilarityScorer

        state.similarity["logistic_similarity"] = LogisticSimilarityScorer(
            joblib.load(art / "models" / "logistic_similarity.joblib"), base, lg)
    emb = manifest.get("components", {}).get("embeddings", {})
    if emb.get("status") == "trained" and _verify(art, manifest, "embeddings/item_attr.npy"):
        from app.recommendation.similarity.embedding import EmbeddingSimilarityScorer, HybridSimilarityScorer

        es = EmbeddingSimilarityScorer(np.load(art / "embeddings" / "item_attr.npy"),
                                       np.load(art / "embeddings" / "item_attr_mask.npy"), base, emb.get("info", {}))
        state.similarity[es.scorer_id] = es
        state.similarity["tag_semantic_hybrid"] = HybridSimilarityScorer(es, base,
                                                                         float(sim_cfg.get("hybrid_tag_weight", 0.6)))
    alternatives = ", ".join(k for k in state.similarity if k != "weighted_attributes") or "none"
    detail = f"weighted_attributes (default); alternatives: {alternatives}"
    if "logistic_similarity" not in state.similarity and lg.get("reason"):
        detail += f". logistic_similarity is {lg.get('status', 'unavailable')}: {lg['reason']}"
    state.set_cap("similar", "ready", detail)

    # ---------------------------------------------------------------- collection recommendations (algorithm 2)
    from app.recommendation.algorithm2.service import build_service

    cc = rec.get("collection", {})
    try:
        svc = build_service(snap, state.constraints, cc, PROJECT_ROOT,
                            lambda m: log_event(log, "algorithm2 bundle", detail=m[:300]))
        state.algorithm2 = svc
        state.set_cap("collection_scorer", "ready" if svc.status == "ready" else "degraded", svc.detail,
                      default=svc.default, catalog_hash=svc.cat.catalog_hash[:12])
        if "C0" in svc.scorers:
            state.set_cap("two_tower", "ready", "C0 is the collection default" if svc.default == "C0" else
                          f"C0 loaded as an alternative (scorer=C0); the frozen comparison selected {svc.default}")
        else:
            state.set_cap("two_tower", "not_configured", "no frozen algorithm-2 run with a two-tower model is configured "
                          "(recommendation.collection.algorithm2_run)")
    except Exception as exc:  # pragma: no cover - recorded; the rest of the app keeps working
        log_event(log, "algorithm2 service failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        state.set_cap("collection_scorer", "failed", f"collection recommendations unavailable: {type(exc).__name__}")

    # ---------------------------------------------------------------- dense retrieval
    idx_dir = art / "index"
    info_path = idx_dir / "dense_info.json"
    if info_path.exists() and _verify(art, manifest, "index/doc_vectors.npy"):
        info = json.loads(info_path.read_text())
        docs = pd.read_parquet(idx_dir / "documents.parquet", columns=["doc_id", "item_key", "kind"])
        encoder, reason = None, ""
        try:
            if info.get("provider") == "bge":
                from app.retrieval.embeddings import BGEEncoder

                encoder = BGEEncoder(info["model"], local_files_only=True)
            elif info.get("provider") == "lsa":
                from app.retrieval.embeddings import LSAEncoder

                encoder = LSAEncoder.load(idx_dir / "lsa.joblib")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:160]}"
        if encoder is not None:
            from app.retrieval.dense import DenseIndex

            state.dense = DenseIndex(np.load(idx_dir / "doc_vectors.npy"), docs.doc_id.tolist(),
                                     docs.item_key.tolist(), docs.kind.tolist(), encoder, info)
            state.set_cap("retrieval_dense", "ready", f"{info.get('provider')} {info.get('model')} dim={info.get('dim')}")
        else:
            state.set_cap("retrieval_dense", "degraded", "query encoder unavailable; lexical retrieval only. " + reason)
    else:
        state.set_cap("retrieval_dense", "not_configured", "no dense vectors for this version; lexical retrieval only")
    state.set_cap("retrieval_lexical", "ready", "SQLite FTS5 / BM25")

    # ---------------------------------------------------------------- images, relations, scenes, llm
    img = manifest.get("index", {}).get("images", {})
    if img.get("status") == "built" and _verify(art, manifest, "index/image_vectors.npy"):
        from app.multimodal.image import ImageGallery, get_image_encoder

        meta = pd.read_parquet(idx_dir / "image_meta.parquet").to_dict("records")
        state.images = ImageGallery(np.load(idx_dir / "image_vectors.npy"), meta,
                                    get_image_encoder(settings.images), img.get("source", "unknown"),
                                    settings.images.get("accept_threshold"))
        if img.get("source") == "synthetic_fixture":
            state.set_cap("screenshot", "degraded", "Synthetic fixture frames only: validates the upload -> "
                          "retrieval pipeline, not real screenshot recognition.", frames=img.get("frames"))
        else:
            state.set_cap("screenshot", "ready", f"{img.get('frames')} frames indexed (uncalibrated threshold)")
    else:
        state.set_cap("screenshot", "limited_by_data", "No frame-level images are available. Cover art is not used "
                      "as a substitute. Provide data/enriched/scene_images/manifest.jsonl to enable.")
    with engine.connect() as conn:
        n_rel = conn.execute(select(func.count()).select_from(M.Relation).where(M.Relation.to_item.is_not(None))).scalar()
        n_from = conn.execute(select(func.count(func.distinct(M.Relation.from_item)))).scalar()
        n_scene = conn.execute(select(func.count()).select_from(M.Scene)).scalar()
    n_served = int(snap.in_catalog.sum()) if snap is not None else 0
    if not n_rel:
        state.set_cap("relations", "limited_by_data",
                      "No relation data for this version (run the optional Jikan enrichment or provide "
                      "data/enriched/relations.jsonl).")
    elif state.dataset_mode == "real" and n_served and n_from / n_served < 0.5:
        # partial coverage: for titles without fetched relation data, a missing edge means "unknown", not "none"
        state.set_cap("relations", "degraded",
                      f"{n_rel} confirmed title-to-title relations from {n_from} of {n_served} titles (partial "
                      "enrichment); for other titles relations are unknown, not absent.",
                      titles_with_relation_data=n_from, served_titles=n_served)
    else:
        state.set_cap("relations", "ready", f"{n_rel} confirmed title-to-title relations from {n_from} titles")
    if n_scene:
        state.set_cap("scenes", "degraded" if settings.mode == "synthetic" else "ready",
                      f"{n_scene} scene records" + (" (fictional synthetic scenes)" if settings.mode == "synthetic" else ""))
    else:
        state.set_cap("scenes", "limited_by_data", "No scene corpus: only synopsis-level plot matching is possible; "
                      "episode numbers are never guessed.")
    llm = settings.llm
    if llm.provider == "none":
        state.set_cap("llm", "not_configured", "No LLM configured: chat uses deterministic parsing and local "
                      "evidence templates (local mode).")
    elif not settings.llm_api_key:
        state.set_cap("llm", "limited_by_credentials", f"{llm.provider} adapter configured but ANILOGIC_LLM_API_KEY "
                      "is not set; local mode is used.")
    else:
        state.set_cap("llm", "ready", f"{llm.provider}:{llm.model} (availability checked per request)")
    return state


class StateHolder:
    """Holds the current ServingState and swaps it when a new activation appears."""

    def __init__(self, settings: Settings, check_interval_s: float = 10.0):
        self.settings = settings
        self.check_interval_s = check_interval_s
        self._state: ServingState | None = None
        self._lock = threading.Lock()
        self._last_check = 0.0

    def get(self) -> ServingState:
        st = self._state
        now = time.monotonic()
        if st is None:
            return self.reload()
        if now - self._last_check > self.check_interval_s:
            self._last_check = now
            try:
                active = read_active(get_engine(self.settings.db_url))
            except Exception:
                active = None
            if active and active.get("dataset_version") != st.dataset_version:
                return self.reload()
        return st

    def reload(self) -> ServingState:
        with self._lock:
            new = load_state(self.settings)
            self._state = new
            self._last_check = time.monotonic()
            log_event(log, "serving state loaded", dataset_version=new.dataset_version,
                      capabilities={k: v.status for k, v in new.capabilities.items()})
            return new
