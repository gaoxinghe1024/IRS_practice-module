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
