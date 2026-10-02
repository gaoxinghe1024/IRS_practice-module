"""Algorithm 2 as served by the website: the NumPy evaluator of LightGBM models reproduces native LightGBM
exactly, and the collection API keeps strict results strict (no padding, no silent relaxation, explicit
relaxation with the strict result count, scorer fallback that keeps the user's filters).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from app.core.config import PROJECT_ROOT


@pytest.mark.ml
def test_numpy_tree_reader_reproduces_native_lightgbm_exactly(tmp_path):
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed")
    code = textwrap.dedent(f"""
        import numpy as np, lightgbm as lgb
        rng = np.random.default_rng(3)
        X = rng.random((3000, 4)); X[rng.random(3000) < 0.2, 1] = np.nan     # feature 1: NaN-aware splits
        y = ((X[:, 0] > 0.5) ^ np.isnan(X[:, 1]) ^ (X[:, 2] > 0.7)).astype(int)
        b = lgb.train({{"objective": "binary", "num_leaves": 15, "min_data_in_leaf": 5, "verbose": -1,
                       "num_threads": 1, "deterministic": True}}, lgb.Dataset(X, y), num_boost_round=40)
        b.save_model(r"{tmp_path / 'model.txt'}")
        T = rng.random((2000, 4)); T[rng.random(2000) < 0.3, 1] = np.nan
        T[rng.random(2000) < 0.1, 2] = np.nan                               # never NaN in training: NaN -> 0.0
        root = b.dump_model()["tree_info"][0]["tree_structure"]
        T[:10, root["split_feature"]] = root["threshold"]                   # exactly on a threshold
        np.save(r"{tmp_path / 'T.npy'}", T)
        np.save(r"{tmp_path / 'native.npy'}", b.predict(T, raw_score=True))
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    from app.recommendation.algorithm2.gbdt_tree import TreeEnsemble

    ens = TreeEnsemble.from_file(tmp_path / "model.txt").prepare()
    T, native = np.load(tmp_path / "T.npy"), np.load(tmp_path / "native.npy")
    assert np.array_equal(ens.predict_raw(T), native)                  # bit masks (served)
    assert np.array_equal(ens.predict_levelwise(T), native)            # node by node (reference)
    assert {int(m) for t in ens.trees for m in t.missing_type} >= {0, 2}   # both missing-value rules exercised


def _collection(client, h, n=5):
    keys = [i["item_key"] for i in client.get("/api/v1/anime", params={"sort": "popularity", "page_size": n}).json()["items"]]
    cid = client.post("/api/v1/collections", json={"name": f"alg2 {np.random.default_rng().integers(1e9)}"},
                      headers=h).json()["id"]
    for k in keys:
        client.put(f"/api/v1/collections/{cid}/items/{k}", headers=h)
    return cid, keys


def test_collection_api_strict_short_list_no_match_and_explicit_relaxation(client):
    from tests.conftest import register

    h, _ = register(client, "alg2")
    cid, keys = _collection(client, h)
    ok = client.post(f"/api/v1/collections/{cid}/recommendations", json={}, headers=h).json()
    assert ok["status"] == "ok" and ok["relaxed"] is False and ok["strict_result_count"] is None
    assert not set(keys) & {i["item_key"] for i in ok["items"]} and ok["meta"]["scorer_label"]
    assert all(i["catalog_score"] is not None for i in ok["items"])            # scored pool only
    strict = client.post(f"/api/v1/collections/{cid}/recommendations", json={"max_episodes": 1, "min_score": 9.9},
                         headers=h).json()
    assert strict["status"] == "no_match" and strict["items"] == [] and strict["relaxed"] is False
    opt = strict["relaxation_options"][0]
    assert opt["original"] and set(opt["original"]) <= {"max_episodes", "min_score"}
    body = {"max_episodes": 1, "min_score": 9.9, **{k: v for k, v in opt["change"].items()
                                                   if k in ("max_episodes", "min_score")},
            "relaxation_applied": {"rule_id": opt["rule_id"], "change": opt["change"], "original": opt["original"]}}
    rel = client.post(f"/api/v1/collections/{cid}/recommendations", json=body, headers=h).json()
    assert rel["relaxed"] is True and rel["relaxed_rules"] == [opt["rule_id"]] and rel["strict_result_count"] == 0
    assert rel["items"] and all(i["episodes"] is not None and i["episodes"] <= body["max_episodes"]
                                and i["catalog_score"] >= body["min_score"] for i in rel["items"])
    # a scorer that is not loaded falls back to the formula, says so, and keeps the user's filters
    fb = client.post(f"/api/v1/collections/{cid}/recommendations", json={"scorer": "B1", "max_episodes": 13},
                     headers=h).json()
    assert fb["meta"]["fallback"] and fb["meta"]["algorithm"] == "A0"
    assert all(i["episodes"] is not None and i["episodes"] <= 13 for i in fb["items"])
