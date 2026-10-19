"""The in-memory serving state for the active dataset version.

Everything a request needs is loaded together: the catalogue snapshot, tags,
aliases, training counts, embedding vectors and the algorithm-2 supplement come
from the business database (written by activation, vectors checked against
their recorded checksums); only model weights are read from the artifact bundle
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
from app.db import vectors
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


CATALOG_COLUMNS = ("item_key", "title", "title_english", "score", "episodes", "duration_minutes", "total_minutes",
                   "type", "status", "members", "aired_from", "cover_url", "in_catalog", "is_recommendable",
                   "is_planner_eligible", "is_explicit", "train_interactions")


def read_catalog(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The active catalogue in catalogue-index order, its tags and aliases (all titles, served or not)."""
    cols = [getattr(M.Anime, c) for c in CATALOG_COLUMNS]
    cat = pd.DataFrame(conn.execute(select(*cols).where(M.Anime.catalog_position.is_not(None))
                                    .order_by(M.Anime.catalog_position)).all(), columns=list(CATALOG_COLUMNS))
    positioned = select(M.Anime.item_key).where(M.Anime.catalog_position.is_not(None))
    tags = pd.DataFrame(conn.execute(select(M.AnimeTag.item_key, M.Tag.namespace, M.Tag.value)
                                     .join(M.Tag, M.Tag.id == M.AnimeTag.tag_id)
                                     .where(M.AnimeTag.item_key.in_(positioned))).all(),
                        columns=["item_key", "namespace", "value"])
    aliases = pd.DataFrame(conn.execute(select(M.AnimeAlias.item_key, M.AnimeAlias.alias, M.AnimeAlias.kind)
                                        .where(M.AnimeAlias.item_key.in_(positioned))
                                        .order_by(M.AnimeAlias.id)).all(), columns=["item_key", "alias", "kind"])
    return cat, tags, aliases


def read_supplement(conn, cc: dict, active: dict) -> tuple[dict | None, dict | None]:
    """(MAL id -> {namespace: values}, provenance) for algorithm 2, as stored by activation."""
    info = active.get("supplement")
    if not cc.get("supplement") or info is None:
        return None, None
    table: dict[int, dict[str, list[str]]] = {}
    for mal_id, ns, value in conn.execute(select(M.SupplementTag.mal_id, M.SupplementTag.namespace,
                                                 M.SupplementTag.value)
                                          .order_by(M.SupplementTag.mal_id, M.SupplementTag.namespace,
                                                    M.SupplementTag.position)):
        table.setdefault(mal_id, {"theme": [], "demographic": []}).setdefault(ns, []).append(value)
    if info.get("status") == "missing" or not table:
        return None, info
    return table, {**info, "source": "business database (supplement_tags)"}


def load_state(settings: Settings, attempts: int = 3) -> ServingState:
    """Build the state; rebuilt when an activation lands while it loads (the catalogue and vectors are read in
    several queries, and only an unchanged activation record proves they all belong to one version)."""
    engine = get_engine(settings.db_url)

    def active_record():
        try:
            return read_active(engine)
        except Exception:  # schema missing; _load_state reports it
            return None

    for _ in range(attempts):
        before = active_record()
        state = _load_state(settings)
        if active_record() == before:
            return state
    return state


def _load_state(settings: Settings) -> ServingState:
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
    with engine.connect() as conn:
        cat, tags, aliases = read_catalog(conn)
        vector_meta = {name: vectors.read_set(conn, name) for name in ("item_attr", "documents", "images")}
        supplement = read_supplement(conn, rec.get("collection", {}), active)
    if not len(cat) or (active.get("counts", {}).get("catalog") not in (None, len(cat))):
        state.set_cap("catalog", "artifact_mismatch", "The catalogue in the database does not match the activation "
                      "record. Re-run activation for this dataset version.")
        return state
    snap = build_snapshot(cat, tags, state.dataset_version, settings.mode)
    counts = cat["train_interactions"]
    snap.with_train_interactions(counts.to_numpy(dtype=np.int64) if counts.notna().all() else None)
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
    item_attr = None
    if emb.get("status") == "trained" and vector_meta["item_attr"]:
        try:
            with engine.connect() as conn:
                rows = conn.execute(select(M.AnimeEmbedding.vector, M.AnimeEmbedding.has_vector)
                                    .join(M.Anime, M.Anime.item_key == M.AnimeEmbedding.item_key)
                                    .where(M.Anime.catalog_position.is_not(None))
                                    .order_by(M.Anime.catalog_position)).all()
            item_attr = (vectors.assemble(vector_meta["item_attr"], [r[0] for r in rows], state.dataset_version),
                         np.array([bool(r[1]) for r in rows], dtype=bool))
            if len(item_attr[0]) != snap.n:
                raise vectors.VectorSetMismatch("item_attr rows do not cover the catalogue")
        except vectors.VectorSetMismatch as exc:
            item_attr = None
            log_event(log, "item_attr vectors refused", error=str(exc)[:200])
    if item_attr is not None:
        from app.recommendation.similarity.embedding import EmbeddingSimilarityScorer, HybridSimilarityScorer

        es = EmbeddingSimilarityScorer(item_attr[0], item_attr[1], base, vector_meta["item_attr"]["info"])
        state.similarity[es.scorer_id] = es
        state.similarity["tag_semantic_hybrid"] = HybridSimilarityScorer(es, base,
                                                                         float(sim_cfg.get("hybrid_tag_weight", 0.6)))
    configured = sim_cfg.get("scorer", "weighted_attributes")
    served = configured if configured in state.similarity else "weighted_attributes"
    alternatives = ", ".join(k for k in state.similarity if k != served) or "none"
    detail = f"{served} (default); alternatives: {alternatives}"
    if served != configured:
        detail += f". The configured default {configured} is unavailable for this version; weighted_attributes is served"
    if "logistic_similarity" not in state.similarity and lg.get("reason"):
        detail += f". logistic_similarity is {lg.get('status', 'unavailable')}: {lg['reason']}"
    state.set_cap("similar", "ready" if served == configured else "degraded", detail, default=served)

    # ---------------------------------------------------------------- collection recommendations (algorithm 2)
    from app.recommendation.algorithm2.service import build_service

    cc = rec.get("collection", {})
    try:
        svc = build_service(snap, state.constraints, cc, PROJECT_ROOT,
                            lambda m: log_event(log, "algorithm2 bundle", detail=m[:300]), supplement=supplement)
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
    doc_vectors, reason = None, ""
    if vector_meta["documents"]:
        try:
            with engine.connect() as conn:
                docs = conn.execute(select(M.DocumentEmbedding.vector, M.KnowledgeDocument.doc_id,
                                           M.KnowledgeDocument.item_key, M.KnowledgeDocument.kind)
                                    .join(M.KnowledgeDocument, M.KnowledgeDocument.doc_id == M.DocumentEmbedding.doc_id)
                                    .order_by(M.DocumentEmbedding.position)).all()
            doc_vectors = vectors.assemble(vector_meta["documents"], [r[0] for r in docs], state.dataset_version)
        except vectors.VectorSetMismatch as exc:
            reason = str(exc)[:200]
    if doc_vectors is not None:
        info = vector_meta["documents"]["info"]
        encoder = None
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

            state.dense = DenseIndex(doc_vectors, [r[1] for r in docs], [r[2] for r in docs], [r[3] for r in docs],
                                     encoder, info)
            state.set_cap("retrieval_dense", "ready", f"{info.get('provider')} {info.get('model')} dim={info.get('dim')}")
        else:
            state.set_cap("retrieval_dense", "degraded", "query encoder unavailable; lexical retrieval only. " + reason)
    elif reason:
        state.set_cap("retrieval_dense", "artifact_mismatch", f"document vectors refused ({reason}); lexical "
                      "retrieval only")
    else:
        state.set_cap("retrieval_dense", "not_configured", "no dense vectors for this version; lexical retrieval only")
    state.set_cap("retrieval_lexical", "ready", "SQLite FTS5 / BM25")

    # ---------------------------------------------------------------- images, relations, scenes, llm
    img = manifest.get("index", {}).get("images", {})
    gallery = None
    if img.get("status") == "built" and vector_meta["images"]:
        cols = (M.SceneImage.image_id, M.SceneImage.scene_id, M.SceneImage.item_key, M.SceneImage.path,
                M.SceneImage.sha256, M.SceneImage.episode, M.SceneImage.timestamp_seconds, M.SceneImage.split,
                M.SceneImage.provenance)
        with engine.connect() as conn:
            rows = conn.execute(select(M.SceneImageEmbedding.vector, *cols)
                                .join(M.SceneImage, M.SceneImage.image_id == M.SceneImageEmbedding.image_id)
                                .order_by(M.SceneImageEmbedding.position)).all()
        try:
            gallery = (vectors.assemble(vector_meta["images"], [r[0] for r in rows], state.dataset_version),
                       [dict(zip([c.key for c in cols], r[1:], strict=True)) for r in rows])
        except vectors.VectorSetMismatch as exc:
            log_event(log, "image vectors refused", error=str(exc)[:200])
    if gallery is not None:
        from app.multimodal.image import ImageGallery, get_image_encoder

        state.images = ImageGallery(gallery[0], gallery[1], get_image_encoder(settings.images),
                                    img.get("source", "unknown"), settings.images.get("accept_threshold"))
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
        n_scene_titles = conn.execute(select(func.count(func.distinct(M.Scene.item_key)))).scalar()
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
    if n_scene and settings.mode == "synthetic":
        state.set_cap("scenes", "degraded", f"{n_scene} scene records (fictional synthetic scenes)")
    elif n_scene and n_served and n_scene_titles / n_served < 0.5:
        # partial coverage: a scene question about another title is answered at the plot level, never with an episode
        state.set_cap("scenes", "degraded",
                      f"{n_scene} scene records for {n_scene_titles} of {n_served} titles; other titles have no scene "
                      "data, so a scene question about them is matched at the plot level without an episode number.",
                      titles_with_scenes=n_scene_titles, served_titles=n_served)
    elif n_scene:
        state.set_cap("scenes", "ready", f"{n_scene} scene records for {n_scene_titles} titles")
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
