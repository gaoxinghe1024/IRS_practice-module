"""Load the scorers of a frozen algorithm-2 run (offline test evaluation and the web service).

Every artefact is checked against freeze_manifest.json before use: model file SHA-256, the catalogue
hash the models were trained on, the feature-schema version and the two-tower encoder layout. Any
mismatch raises BundleMismatch; the service then falls back to the formula with a recorded status
(it never silently loads an incompatible model and never relaxes the user's filters).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.recommendation.algorithm2.catalog import Alg2Catalog
from app.recommendation.algorithm2.features import FEATURE_VERSION
from app.recommendation.algorithm2.scorers import (
    FormulaScorer,
    LegacyScorer,
    LightGBMScorer,
    TreeEnsembleScorer,
    TwoTowerScorer,
)


class BundleMismatch(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_freeze(run_dir: Path) -> dict:
    p = Path(run_dir) / "freeze_manifest.json"
    if not p.exists():
        raise BundleMismatch(f"no freeze_manifest.json in {run_dir}")
    return json.loads(p.read_text())


def load_scorers(run_dir: Path, cat: Alg2Catalog, labels: list[str] | None = None, num_threads: int = 1,
                 include_aux: bool = False, families: tuple = ("B0", "B1", "C0"),
                 gbdt_engine: str = "numpy") -> tuple[dict, dict]:
    """Return ({label: scorer}, freeze manifest). Labels: A0, A1, <family>_seed<seed> (e.g. B1_seed42).

    gbdt_engine "numpy" reads the LightGBM text model without the native library (safe next to PyTorch);
    "native" uses lightgbm.Booster (offline, in processes that never import PyTorch).
    """
    run_dir = Path(run_dir)
    fz = read_freeze(run_dir)
    if fz["catalog_hash"] != cat.catalog_hash:
        raise BundleMismatch("catalogue hash differs from the one the models were trained on "
                             f"({fz['catalog_hash'][:12]} != {cat.catalog_hash[:12]})")
    if fz.get("feature_version") != FEATURE_VERSION:
        raise BundleMismatch(f"feature version {fz.get('feature_version')} != {FEATURE_VERSION}")
    out: dict = {"A0": FormulaScorer("A0", fz["formula"]["A0"]["hits_weight"]),
                 "A1": FormulaScorer("A1", fz["formula"]["A1"]["hits_weight"])}
    encoder = None
    for art in fz["models"]:
        label = f"{art['family']}_seed{art['seed']}"
        if (labels is not None and label not in labels) or art["family"] not in families:
            continue
        path = run_dir / art["file"]
        if not path.exists() or _sha256(path) != art["sha256"]:
            raise BundleMismatch(f"{art['file']} is missing or its SHA-256 differs from the freeze manifest")
        meta = {"model_sha256": art["sha256"], "seed": art["seed"], "file": art["file"]}
        if art["family"] in ("B0", "B1"):
            feats = fz["configs"][art["family"]].get("features") or fz["feature_schema"]["names"]
            if gbdt_engine == "native":
                import lightgbm as lgb

                booster = lgb.Booster(model_file=str(path))
                n_feat = booster.num_feature()
                scorer = LightGBMScorer(label, booster, feats, meta, num_threads)
            else:
                from app.recommendation.algorithm2.gbdt_tree import TreeEnsemble

                ens = TreeEnsemble.from_file(path)
                n_feat = ens.num_features
                scorer = TreeEnsembleScorer(label, ens, feats, meta)
            if n_feat != len(feats):
                raise BundleMismatch(f"{label}: model expects {n_feat} features, schema has {len(feats)}")
            out[label] = scorer
        elif art["family"] == "C0":
            from app.recommendation.algorithm2.two_tower import TowerEncoder, load_model

            model, blob = load_model(path)
            if encoder is None:
                encoder = TowerEncoder(cat, int(blob["encoder"]["studio_min_count"]))
            spec = encoder.spec()
            for k in ("version", "vocab_sha256", "d_item", "d_coll", "dims"):
                if spec[k] != blob["encoder"][k]:
                    raise BundleMismatch(f"{label}: encoder {k} differs from the trained model")
            out[label] = TwoTowerScorer(label, model, encoder, meta)
    if include_aux:
        out["AUX_slides_v0"] = LegacyScorer("AUX_slides_v0")
        out["AUX_profile_v1"] = LegacyScorer("AUX_profile_v1")
    return out, fz
