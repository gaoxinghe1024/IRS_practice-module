"""Activate one dataset version: verify artifacts, load everything the running
system reads except model weights into the business database in ONE
transaction, rebuild the FTS5 indexes, then atomically switch
artifacts/<mode>/active.json.

Loaded: the full catalogue in catalogue-index order (anime.catalog_position;
unserved titles with in_catalog = 0) with its train-split rating counts, tags
and aliases; relations, characters, scenes and knowledge documents of served
titles; the attribute, document and frame embedding vectors; and, in real mode,
the algorithm-2 supplementary theme / demographic table. Serving reads these
from the database and only model files (models/, the LSA encoder) from the
artifact bundle.

A running app notices the new active version (system_state) and swaps its
in-memory serving state as a whole, so a request never mixes a new index
with an old ID mapping. Existing user data is never deleted: titles that
disappear from the new version but are referenced by users stay in the
database, hidden from the catalogue (in_catalog = 0).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.config import Settings
from app.core.text import norm, norm_title
from app.db import models as M
from app.db import vectors
from app.db.session import get_engine, run_migrations
from app.recommendation.algorithm2.catalog import SUPPLEMENT_NAMESPACES, file_sha256, load_supplement
from pipelines.common import latest_dataset_version, read_json, rel, sha256_file, utc_now_iso, write_json
from pipelines.fit import artifact_dir
from pipelines.processed_io import ProcessedDataset


class ActivationError(RuntimeError):
    pass


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "tolist") and getattr(v, "ndim", 0) > 0:   # parquet list columns -> numpy arrays
        return v.tolist()
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def verify_artifacts(settings: Settings, version: str) -> dict:
    art = artifact_dir(settings, version)
    mpath = art / "manifest.json"
    if not mpath.exists():
        raise ActivationError(f"artifact manifest missing: {rel(mpath)} (run train)")
    manifest = read_json(mpath)
    if manifest.get("dataset_version") != version:
        raise ActivationError("artifact manifest belongs to a different dataset version")
    if "index" not in manifest:
        raise ActivationError("index not built for this version (run build-index)")
    problems = []
    for group in (manifest.get("files", {}), manifest["index"].get("files", {})):
        for relpath, digest in group.items():
            p = art / relpath
            if not p.exists():
                problems.append(f"missing {relpath}")
            elif sha256_file(p) != digest:
                problems.append(f"checksum mismatch {relpath}")
    if problems:
        raise ActivationError("; ".join(problems[:10]))
    return manifest


def supplement_rows(settings: Settings) -> tuple[list[dict], dict | None]:
    """Rows of the algorithm-2 supplement table (real mode only) and the provenance stored with them."""
    path = settings.recommendation.get("collection", {}).get("supplement")
    if settings.mode != "real" or not path:
        return [], None
    p = settings._abs(path)
    if not p.exists():
        return [], {"path": str(path), "status": "missing"}
    table = load_supplement(p)
    rows = [{"mal_id": mal_id, "namespace": ns, "position": j, "value": v}
            for mal_id, r in table.items() for ns in SUPPLEMENT_NAMESPACES for j, v in enumerate(r[ns])]
    return rows, {"path": str(path), "sha256": file_sha256(p), "namespaces": list(SUPPLEMENT_NAMESPACES),
                  "rule": "union with snapshot values; MAL id join", "titles": len(table)}


def activate(settings: Settings, version: str | None = None, log=print) -> dict[str, Any]:
    version = version or latest_dataset_version(settings)
    manifest = verify_artifacts(settings, version)
    art = artifact_dir(settings, version)
    ds = ProcessedDataset.for_version(settings, version)
    run_migrations(settings.db_url)
    engine = get_engine(settings.db_url)

    catalog = pd.read_parquet(art / "catalog" / "catalog.parquet").reset_index(drop=True)
    for col in ("provenance", "missing_fields", "exclusion_reasons"):
        if col in catalog:
            catalog[col] = catalog[col].map(json.loads)
    files = manifest.get("files", {})
    index_files = manifest["index"].get("files", {})
    if "stats/train_interactions.npy" in files:
        counts = np.load(art / "stats" / "train_interactions.npy")
        catalog["train_interactions"] = [int(c) for c in counts] if len(counts) == len(catalog) else None
    catalog["catalog_position"] = np.arange(len(catalog))
    served = catalog[catalog["in_catalog"].astype(bool)]
    tags = pd.read_parquet(art / "catalog" / "item_tags.parquet")
    aliases = pd.read_parquet(art / "catalog" / "aliases.parquet")
    docs = pd.read_parquet(art / "index" / "documents.parquet")
    relations = ds.table("relations")
    characters = ds.table("characters")
    scenes = ds.table("scenes")
    scene_images = ds.table("scene_images")
    served_keys = set(served["item_key"])
    catalog_keys = set(catalog["item_key"])
    supp_rows, supp_info = supplement_rows(settings)

    anime_cols = [c.name for c in M.Anime.__table__.columns]
    rows = []
    for r in catalog.to_dict("records"):
        row = {c: _clean(r.get(c)) for c in anime_cols if c in r}
        row["dataset_version"] = version
        for c in ("episodes", "mal_id", "scored_by", "rank", "popularity", "members", "favorites", "premiered_year",
                  "catalog_position", "train_interactions"):
            if row.get(c) is not None:
                row[c] = int(row[c])
        rows.append(row)
    served_rows = [r for r in rows if r["item_key"] in served_keys]

    with engine.begin() as conn:
        # ------------------------------------------------------------- anime (whole catalogue, catalogue order)
        conn.execute(update(M.Anime).values(catalog_position=None))   # positions move between versions
        for start in range(0, len(rows), 400):  # 400 rows x ~37 cols < SQLite 32766 params
            chunk = rows[start:start + 400]
            stmt = sqlite_insert(M.Anime.__table__).values(chunk)
            upd = {c: stmt.excluded[c] for c in chunk[0] if c != "item_key"}
            conn.execute(stmt.on_conflict_do_update(index_elements=["item_key"], set_=upd))
        existing = set(conn.execute(select(M.Anime.item_key)).scalars())
        gone = existing - catalog_keys
        if gone:
            referenced = set()
            for _tbl, col in ((M.Rating, M.Rating.item_key), (M.CollectionItem, M.CollectionItem.item_key),
                             (M.WatchProgress, M.WatchProgress.item_key), (M.Exclusion, M.Exclusion.item_key),
                             (M.PlanItem, M.PlanItem.item_key), (M.FeedbackEvent, M.FeedbackEvent.item_key)):
                referenced |= set(conn.execute(select(col).distinct()).scalars())
            hide = gone & referenced
            if hide:
                conn.execute(update(M.Anime).where(M.Anime.item_key.in_(list(hide))).values(in_catalog=False))
            drop = list(gone - referenced)
            for s in range(0, len(drop), 500):
                conn.execute(delete(M.Anime).where(M.Anime.item_key.in_(drop[s:s + 500])))
        # ------------------------------------------------------------- tags
        tags = tags[tags["item_key"].isin(catalog_keys)].copy()
        tags["normalized"] = tags["value"].map(norm)
        uniq = tags.drop_duplicates(["namespace", "normalized"])[["namespace", "value", "normalized"]]
        if len(uniq):
            stmt = sqlite_insert(M.Tag.__table__)
            for s in range(0, len(uniq), 2000):
                conn.execute(stmt.on_conflict_do_nothing(index_elements=["namespace", "normalized"]),
                             uniq.iloc[s:s + 2000].to_dict("records"))
        tag_ids = {(ns, n): i for i, ns, n in conn.execute(select(M.Tag.id, M.Tag.namespace, M.Tag.normalized))}
        conn.execute(delete(M.AnimeTag))
        at = tags.assign(tag_id=[tag_ids[(a, b)] for a, b in zip(tags.namespace, tags.normalized, strict=True)])
        at = at.drop_duplicates(["item_key", "tag_id"])[["item_key", "tag_id", "source"]]
        for s in range(0, len(at), 5000):
            conn.execute(M.AnimeTag.__table__.insert(), at.iloc[s:s + 5000].to_dict("records"))
        # ------------------------------------------------------------- aliases
        conn.execute(delete(M.AnimeAlias))
        al = aliases[aliases["item_key"].isin(catalog_keys)].drop_duplicates(["item_key", "alias"]).copy()
        al["normalized"] = al["alias"].map(norm_title)
        for s in range(0, len(al), 5000):
            conn.execute(M.AnimeAlias.__table__.insert(), al.iloc[s:s + 5000][["item_key", "alias", "normalized",
                                                                                 "kind"]].to_dict("records"))
        # ------------------------------------------------------------- relations / characters
        conn.execute(delete(M.Relation))
        if len(relations):
            rel_rows = []
            for r in relations.to_dict("records"):
                if r["from_item"] not in served_keys:
                    continue
                to_item = _clean(r.get("to_item"))
                if to_item is not None and to_item not in served_keys:
                    to_item, etype = None, "anime_unresolved"
                else:
                    etype = r["to_entity_type"]
                rel_rows.append({"from_item": r["from_item"], "to_item": to_item, "to_entity_type": etype,
                                 "to_external_id": _clean(r.get("to_external_id")), "to_name": _clean(r.get("to_name")),
                                 "relation_type": r["relation_type"], "status": r.get("status") or "confirmed",
                                 "source_ref": r["source_ref"]})
            if rel_rows:
                conn.execute(M.Relation.__table__.insert(), rel_rows)
        conn.execute(delete(M.AnimeCharacter))
        conn.execute(delete(M.Character))
        if len(characters):
            ch = characters[characters["item_key"].isin(served_keys)]
            uniq_ch = ch.drop_duplicates("source_id")
            if len(uniq_ch):
                conn.execute(M.Character.__table__.insert(),
                             [{"source_id": r["source_id"], "name": r["name"], "provenance": {"source_ref": r["source_ref"]}}
                              for r in uniq_ch.to_dict("records")])
                ids = dict(conn.execute(select(M.Character.source_id, M.Character.id)).all())
                links = ch.drop_duplicates(["item_key", "source_id"])
                conn.execute(M.AnimeCharacter.__table__.insert(),
                             [{"item_key": r["item_key"], "character_id": ids[r["source_id"]], "role": _clean(r["role"]),
                               "source_ref": r["source_ref"]} for r in links.to_dict("records")])
        # ------------------------------------------------------------- scenes
        conn.execute(delete(M.SceneImage))
        conn.execute(delete(M.Scene))
        if len(scenes):
            conn.execute(M.Scene.__table__.insert(), [
                {k: _clean(v) for k, v in r.items()} | {"characters": _clean(r.get("characters")) or []}
                for r in scenes.to_dict("records") if r["item_key"] in served_keys])
        image_rows = [
            {"image_id": r["image_id"], "scene_id": _clean(r.get("scene_id")), "item_key": r["item_key"],
             "path": r["path"], "sha256": r["sha256"], "episode": _clean(r.get("episode")),
             "timestamp_seconds": _clean(r.get("timestamp_seconds")), "split": r.get("split", "gallery"),
             "provenance": r.get("provenance") or {}}
            for r in scene_images.to_dict("records") if r["item_key"] in served_keys] if len(scene_images) else []
        if image_rows:
            conn.execute(M.SceneImage.__table__.insert(), image_rows)
        # ------------------------------------------------------------- documents + FTS
        conn.execute(delete(M.KnowledgeDocument))
        doc_rows, doc_pos = [], []
        for pos, r in enumerate(docs.to_dict("records")):
            if r["item_key"] not in served_keys:
                continue
            doc_pos.append(pos)
            doc_rows.append({"doc_id": r["doc_id"], "item_key": r["item_key"], "kind": r["kind"], "title": r["title"],
                             "body": r["body"], "spoiler_level": r["spoiler_level"], "language": r["language"],
                             "source_type": r["source_type"], "source_ref": r["source_ref"],
                             "source_url": _clean(r.get("source_url")), "locator": json.loads(r["locator"]),
                             "retrieved_at": _clean(r.get("retrieved_at")), "content_hash": r["content_hash"],
                             "dataset_version": version})
        for s in range(0, len(doc_rows), 2000):
            conn.execute(M.KnowledgeDocument.__table__.insert(), doc_rows[s:s + 2000])
        conn.execute(text("DELETE FROM anime_fts"))
        alias_text = al.groupby("item_key")["alias"].apply(lambda s: " | ".join(s)).to_dict()
        fts_rows = [{"k": r["item_key"], "t": r["title"], "a": alias_text.get(r["item_key"], "")} for r in served_rows]
        conn.execute(text("INSERT INTO anime_fts(item_key, title, aliases) VALUES (:k, :t, :a)"), fts_rows)
        conn.execute(text("DELETE FROM docs_fts"))
        conn.execute(text("INSERT INTO docs_fts(doc_id, item_key, kind, title, body) VALUES (:d, :i, :k, :t, :b)"),
                     [{"d": d["doc_id"], "i": d["item_key"], "k": d["kind"], "t": d["title"], "b": d["body"]}
                      for d in doc_rows])
        # ------------------------------------------------------------- embedding vectors
        sets = {}
        conn.execute(delete(M.AnimeEmbedding))
        emb = manifest.get("components", {}).get("embeddings", {})
        if emb.get("status") == "trained" and "embeddings/item_attr.npy" in files:
            mat = np.load(art / "embeddings" / "item_attr.npy")
            mask = np.load(art / "embeddings" / "item_attr_mask.npy")
            if len(mat) != len(rows) or len(mask) != len(rows):
                raise ActivationError("item_attr embeddings are not aligned with the catalogue")
            blobs = vectors.row_blobs(mat)
            emb_rows = [{"item_key": r["item_key"], "vector": blobs[i], "has_vector": bool(mask[i])}
                        for i, r in enumerate(rows)]
            for s in range(0, len(emb_rows), 2000):
                conn.execute(M.AnimeEmbedding.__table__.insert(), emb_rows[s:s + 2000])
            sets["item_attr"] = vectors.record_set(conn, "item_attr", version, mat, emb.get("info", {}))
        else:
            vectors.drop_set(conn, "item_attr")
        conn.execute(delete(M.DocumentEmbedding))
        if "index/doc_vectors.npy" in index_files and (art / "index" / "dense_info.json").exists():
            mat = np.load(art / "index" / "doc_vectors.npy")
            if len(mat) != len(docs):
                raise ActivationError("document vectors are not aligned with documents.parquet")
            mat = mat[doc_pos]
            blobs = vectors.row_blobs(mat)
            for s in range(0, len(doc_rows), 2000):
                conn.execute(M.DocumentEmbedding.__table__.insert(),
                             [{"doc_id": d["doc_id"], "position": s + j, "vector": blobs[s + j]}
                              for j, d in enumerate(doc_rows[s:s + 2000])])
            sets["documents"] = vectors.record_set(conn, "documents", version, mat,
                                                   read_json(art / "index" / "dense_info.json"))
        else:
            vectors.drop_set(conn, "documents")
        conn.execute(delete(M.SceneImageEmbedding))
        img = manifest["index"].get("images", {})
        if img.get("status") == "built" and "index/image_vectors.npy" in index_files:
            mat = np.load(art / "index" / "image_vectors.npy")
            meta = pd.read_parquet(art / "index" / "image_meta.parquet")
            if len(mat) != len(meta):
                raise ActivationError("image vectors are not aligned with image_meta.parquet")
            stored = {r["image_id"] for r in image_rows}
            keep = [i for i, image_id in enumerate(meta["image_id"]) if image_id in stored]
            mat = mat[keep]
            if keep:
                conn.execute(M.SceneImageEmbedding.__table__.insert(),
                             [{"image_id": meta["image_id"].iloc[i], "position": j, "vector": mat[j].tobytes()}
                              for j, i in enumerate(keep)])
            sets["images"] = vectors.record_set(conn, "images", version, mat, img)
        else:
            vectors.drop_set(conn, "images")
        # ------------------------------------------------------------- algorithm-2 supplement
        conn.execute(delete(M.SupplementTag))
        for s in range(0, len(supp_rows), 5000):
            conn.execute(M.SupplementTag.__table__.insert(), supp_rows[s:s + 5000])
        # ------------------------------------------------------------- state
        state = {"dataset_mode": settings.mode, "dataset_version": version, "artifact_dir": rel(art),
                 "manifest_sha256": sha256_file(art / "manifest.json"), "activated_at": utc_now_iso(),
                 "counts": {"anime": len(served_rows), "catalog": len(rows), "documents": len(doc_rows),
                            "tags": int(len(at)), "aliases": int(len(al)),
                            "vectors": {k: v["count"] for k, v in sets.items()},
                            "supplement_values": len(supp_rows)},
                 "supplement": supp_info}
        conn.execute(sqlite_insert(M.SystemState.__table__).values(key="active_dataset", value=state)
                     .on_conflict_do_update(index_elements=["key"], set_={"value": state}))
        conn.execute(update(M.ArtifactRecord).where(M.ArtifactRecord.dataset_mode == settings.mode)
                     .values(is_active=False))
        conn.execute(sqlite_insert(M.ArtifactRecord.__table__).values(
            dataset_mode=settings.mode, version=version, kind="bundle", path=rel(art), manifest=manifest,
            checksum=state["manifest_sha256"], is_active=True).on_conflict_do_update(
            index_elements=["dataset_mode", "version", "kind"],
            set_={"manifest": manifest, "checksum": state["manifest_sha256"], "is_active": True}))
    write_json(settings.artifacts_dir / "active.json", state)
    log(f"[activate] {version}: {len(served_rows)} served of {len(rows)} titles, {len(doc_rows)} documents, "
        f"vectors {state['counts']['vectors']} -> {rel(settings.db_path)}")
    return state
