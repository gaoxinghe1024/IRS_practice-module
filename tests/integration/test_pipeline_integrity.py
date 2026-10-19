import json

import pytest

from app.services.serving import load_state
from pipelines.activate import ActivationError, activate, verify_artifacts
from pipelines.common import latest_dataset_version, read_json
from pipelines.fit import artifact_dir


def test_activation_is_idempotent_and_state_matches(synthetic_env):
    s = synthetic_env
    v = latest_dataset_version(s)
    first = activate(s, v)
    second = activate(s, v)
    assert first["counts"] == second["counts"] and first["dataset_version"] == v
    st = load_state(s)
    assert st.dataset_version == v and st.cap("catalog").status == "ready"
    assert st.cap("screenshot").status == "degraded"          # synthetic fixture frames only
    assert st.cap("llm").status == "not_configured"


def test_tampered_artifact_is_refused(synthetic_env):
    s = synthetic_env
    v = latest_dataset_version(s)
    art = artifact_dir(s, v)
    target = art / "models" / "icf.npz"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"tamper")
        with pytest.raises(ActivationError, match="checksum mismatch"):
            verify_artifacts(s, v)
    finally:
        target.write_bytes(original)
    verify_artifacts(s, v)


def test_manifest_mismatch_degrades_serving(synthetic_env):
    s = synthetic_env
    art = artifact_dir(s, latest_dataset_version(s))
    mpath = art / "manifest.json"
    original = mpath.read_text()
    try:
        m = json.loads(original)
        m["tampered"] = True
        mpath.write_text(json.dumps(m))
        st = load_state(s)
        assert st.cap("catalog").status == "artifact_mismatch" and st.snap is None
    finally:
        mpath.write_text(original)
    assert load_state(s).cap("catalog").status == "ready"


def test_manifests_record_versions_and_hashes(synthetic_env):
    s = synthetic_env
    v = latest_dataset_version(s)
    m = read_json(artifact_dir(s, v) / "manifest.json")
    assert m["dataset_mode"] == "synthetic" and m["split"]["fingerprint"]
    assert m["components"]["icf"]["status"] == "trained" and m["components"]["icf"]["sha256"]
    assert m["index"]["dense"]["provider"] == "lsa" and m["index"]["dense"]["doc_ids_sha256"]
    assert "python" in m["dependencies"]
    assert all("\\" not in name for group in (m["files"], m["index"]["files"]) for name in group)


def test_synthetic_generation_is_reproducible(synthetic_env, tmp_path):
    from app.core.config import load_settings
    from pipelines.synthetic import generate

    env = {k: v for k, v in __import__("os").environ.items()}
    env["ANILOGIC_PATHS__DATA_ROOT"] = str(tmp_path / "data")
    s2 = load_settings("synthetic", environ=env)
    a = generate(s2, log=lambda *_: None)["files"]
    b = generate(s2, log=lambda *_: None)["files"]
    assert a == b


def test_similar_default_comes_from_config_with_a_reported_fallback(synthetic_env):
    s = synthetic_env.model_copy(deep=True)
    s.recommendation["similar"]["scorer"] = "no_such_scorer"
    cap = load_state(s).cap("similar")
    assert cap.status == "degraded" and "no_such_scorer is unavailable" in cap.detail
    assert cap.detail.startswith("algorithm1_a (default)")
    st = load_state(synthetic_env)
    if "logistic_similarity" in st.similarity:                       # trained on the synthetic labels
        s.recommendation["similar"]["scorer"] = "logistic_similarity"
        cap = load_state(s).cap("similar")
        assert cap.status == "ready" and cap.detail.startswith("logistic_similarity (default)")


def test_serving_reads_catalogue_and_vectors_from_the_database(synthetic_env):
    """The snapshot, tag matrices, counts and vectors served from SQLite equal the activated artifact files, and
    serving still works with those files moved away (only model weights are read from the bundle)."""
    import shutil

    import numpy as np
    import pandas as pd

    from app.recommendation.features import build_snapshot

    s = synthetic_env
    v = latest_dataset_version(s)
    art = artifact_dir(s, v)
    st = load_state(s)
    ref = build_snapshot(pd.read_parquet(art / "catalog" / "catalog.parquet"),
                         pd.read_parquet(art / "catalog" / "item_tags.parquet"), v, "synthetic")
    snap = st.snap
    assert list(snap.keys) == list(ref.keys) and snap.titles == ref.titles
    for name in ("score", "episodes", "duration", "total_minutes", "members", "year"):
        np.testing.assert_array_equal(getattr(snap, name), getattr(ref, name))
    for name in ("in_catalog", "recommendable", "planner_eligible", "explicit"):
        np.testing.assert_array_equal(getattr(snap, name), getattr(ref, name))
    for ns, tm in ref.tags.items():
        assert snap.tags[ns].vocab == tm.vocab and (snap.tags[ns].X != tm.X).nnz == 0
    np.testing.assert_array_equal(snap.train_interactions, np.load(art / "stats" / "train_interactions.npy"))
    if read_json(art / "manifest.json")["components"].get("embeddings", {}).get("status") == "trained":
        emb = next(sc for k, sc in st.similarity.items() if k.endswith("_attribute_embedding"))
        np.testing.assert_array_equal(emb.vectors, np.load(art / "embeddings" / "item_attr.npy"))
        np.testing.assert_array_equal(emb.has_vector, np.load(art / "embeddings" / "item_attr_mask.npy"))
    np.testing.assert_array_equal(st.dense.vectors, np.load(art / "index" / "doc_vectors.npy"))
    assert st.dense.doc_ids == pd.read_parquet(art / "index" / "documents.parquet")["doc_id"].tolist()
    np.testing.assert_array_equal(st.images.vectors, np.load(art / "index" / "image_vectors.npy"))

    moved = art.parent / f"{v}-moved"
    moved.mkdir()
    names = tuple(n for n in ("catalog", "embeddings", "stats", "index/doc_vectors.npy", "index/documents.parquet",
                              "index/image_vectors.npy", "index/image_meta.parquet", "index/dense_info.json")
                  if (art / n).exists())
    try:
        for n in names:
            (moved / n).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(art / n, moved / n)
        st2 = load_state(s)
        assert st2.cap("catalog").status == "ready" and list(st2.snap.keys) == list(ref.keys)
        assert st2.cap("retrieval_dense").status == "ready" and st2.images is not None
        assert len(st2.similarity) == len(st.similarity)
    finally:
        for n in names:
            if (moved / n).exists():
                shutil.move(moved / n, art / n)
        shutil.rmtree(moved)


def test_tampered_vectors_in_the_database_are_refused(synthetic_env):
    from sqlalchemy import select, update

    from app.db import models as M
    from app.db.session import get_engine

    s = synthetic_env
    engine = get_engine(s.db_url)
    with engine.connect() as conn:
        doc_id, original = conn.execute(select(M.DocumentEmbedding.doc_id, M.DocumentEmbedding.vector)
                                        .order_by(M.DocumentEmbedding.position)).first()
    try:
        with engine.begin() as conn:
            conn.execute(update(M.DocumentEmbedding).where(M.DocumentEmbedding.doc_id == doc_id)
                         .values(vector=bytes(len(original))))
        st = load_state(s)
        assert st.cap("retrieval_dense").status == "artifact_mismatch" and st.dense is None
        assert st.cap("catalog").status == "ready"
    finally:
        with engine.begin() as conn:
            conn.execute(update(M.DocumentEmbedding).where(M.DocumentEmbedding.doc_id == doc_id)
                         .values(vector=original))
    assert load_state(s).cap("retrieval_dense").status == "ready"
