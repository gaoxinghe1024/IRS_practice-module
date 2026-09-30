"""Build the retrieval index artifacts for one dataset version.

artifacts/<mode>/<version>/index/
    documents.parquet      every knowledge document (rows aligned with doc_vectors.npy)
    doc_vectors.npy        L2-normalised dense vectors (BGE when available, else LSA)
    dense_info.json        provider / model / revision / dim / doc_ids hash
    lsa.joblib             only for the LSA fallback encoder
    image_vectors.npy      scene-frame gallery vectors (only when frames exist)
    image_meta.parquet

The lexical (FTS5) index is (re)built inside the business database at activation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.text import approx_tokens, sha256_text
from app.retrieval.documents import build_documents
from app.retrieval.embeddings import BGEEncoder, LSAEncoder, choose_provider
from pipelines.common import latest_dataset_version, read_json, read_jsonl, rel, sha256_file, utc_now_iso, write_json
from pipelines.fit import _device, artifact_dir
from pipelines.processed_io import ProcessedDataset


def _load_reviews(settings: Settings) -> list[dict]:
    p = settings._abs(settings.real.get("enriched_dir", "data/enriched")) / "reviews.jsonl" \
        if settings.mode == "real" else None
    if p is None or not p.exists():
        return []
    out = []
    for r in read_jsonl(p):
        if r.get("mal_id") and r.get("text") and r.get("review_id"):
            out.append({"item_key": f"mal:{int(r['mal_id'])}", "text": r["text"], "review_id": str(r["review_id"]),
                        "source_ref": r.get("source_ref", "data/enriched/reviews.jsonl")})
    return out


def build_index(settings: Settings, version: str | None = None, log=print) -> dict[str, Any]:
    t0 = time.time()
    version = version or latest_dataset_version(settings)
    ds = ProcessedDataset.for_version(settings, version)
    out = artifact_dir(settings, version)
    idx_dir = out / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    cfg = settings.retrieval
    provider = choose_provider(cfg.get("embedding", {}))
    encoder = None
    if provider == "bge":
        try:
            encoder = BGEEncoder(cfg["embedding"].get("model", "BAAI/bge-small-en-v1.5"),
                                 batch_size=int(cfg["embedding"].get("batch_size", 64)), device=_device(settings))
            counter = encoder.count_tokens
        except Exception as exc:
            log(f"[index] BGE unavailable ({type(exc).__name__}); falling back to LSA")
            provider, encoder = "lsa", None
    if encoder is None:
        counter = approx_tokens
    docs = build_documents(ds.catalog(), ds.table("item_tags"), ds.table("relations"), ds.table("characters"),
                           ds.table("scenes"), _load_reviews(settings), counter, cfg)
    df = pd.DataFrame([d.to_row() for d in docs])
    df["dataset_version"] = version
    df["locator"] = df["locator"].map(lambda v: __import__("json").dumps(v, default=str))
    df.to_parquet(idx_dir / "documents.parquet", index=False)
    log(f"[index] {len(docs)} documents ({df.kind.value_counts().to_dict()})")
    texts = [d.dense_text for d in docs]
    t1 = time.time()
    if provider == "bge":
        vecs = encoder.encode_documents(texts)
        info = encoder.info()
    elif provider == "lsa":
        lsa = LSAEncoder(int(cfg.get("embedding", {}).get("lsa_dim", 256))).fit(texts)
        vecs = lsa.encode_documents(texts)
        lsa.save(idx_dir / "lsa.joblib")
        info = lsa.info()
    else:
        vecs, info = None, {"provider": "none"}
    if vecs is not None:
        np.save(idx_dir / "doc_vectors.npy", vecs.astype(np.float32))
    info.update({"doc_count": len(docs), "doc_ids_sha256": sha256_text("\n".join(df.doc_id)),
                 "encode_seconds": round(time.time() - t1, 1), "device": _device(settings) if provider == "bge" else "cpu",
                 "created_at": utc_now_iso()})
    write_json(idx_dir / "dense_info.json", info)
    log(f"[index] dense vectors: {info.get('provider')} {info.get('model')} dim={info.get('dim')} "
        f"in {info['encode_seconds']}s")
    image_info = _build_image_index(settings, ds, idx_dir, log)
    manifest_path = out / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"dataset_version": version}
    manifest["index"] = {"documents": len(docs), "kinds": df.kind.value_counts().to_dict(), "dense": info,
                         "images": image_info, "built_at": utc_now_iso(), "seconds": round(time.time() - t0, 1),
                         "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(idx_dir.rglob("*"))
                                   if p.is_file()}}
    write_json(manifest_path, manifest)
    log(f"[index] done in {manifest['index']['seconds']}s -> {rel(idx_dir)}")
    return manifest["index"]


def _build_image_index(settings: Settings, ds: ProcessedDataset, idx_dir, log) -> dict:
    from app.multimodal.image import get_image_encoder

    meta = ds.table("scene_images")
    if not len(meta):
        return {"status": "limited_by_data", "reason": "no frame-level images (covers are not screenshots)"}
    enc = get_image_encoder(settings.images)
    vecs, ok_rows = [], []
    for row in meta.to_dict("records"):
        path = settings.data_root.parent / row["path"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            continue
        vecs.append(enc.encode_path(path))
        ok_rows.append(row)
    if not vecs:
        return {"status": "limited_by_data", "reason": "frame manifest present but files missing / hash mismatch"}
    np.save(idx_dir / "image_vectors.npy", np.vstack(vecs).astype(np.float32))
    pd.DataFrame(ok_rows).to_parquet(idx_dir / "image_meta.parquet", index=False)
    source = "synthetic_fixture" if settings.mode == "synthetic" else "frame_manifest"
    log(f"[index] image gallery: {len(ok_rows)} frames via {enc.info()['encoder']} ({source})")
    return {"status": "built", "frames": len(ok_rows), "encoder": enc.info(), "source": source}
