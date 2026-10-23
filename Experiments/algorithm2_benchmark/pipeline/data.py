"""Data adapter and `prepare` stage for the algorithm-2 experiments.

real     the processed Kaggle MAL 2023 dataset (data/processed/real/<version>/) plus the theme /
         demographic supplement; personal ratings >= threshold on catalogue members form each offline
         user's positive preference set (a proxy for a collection; there is no real collection data).
fixture  a tiny generated dataset (clearly labelled synthetic/fixture) that only checks the logic.

Interaction cleaning (counts are recorded in data_manifest.json):
  * ratings outside 1-10, unknown item indices and items outside the catalogue are dropped;
  * duplicate (user, item) rows: identical ones are de-duplicated; conflicting ratings (no timestamp to
    order them) exclude the pair. The processed dataset already de-duplicated (0 conflicts recorded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm2_benchmark.pipeline.common import now, read_json, sha256_df, sha256_file, write_json
from app.core.config import PROJECT_ROOT, load_settings
from app.recommendation.algorithm2.catalog import Alg2Catalog, build_catalog, catalog_from_settings
from app.recommendation.constraints import ConstraintEngine, build_engine
from app.recommendation.features import build_snapshot, snapshot_from_processed

PREPARE_VERSION = "alg2-prepare-v1"


@dataclass
class Prepared:
    cat: Alg2Catalog
    engine: ConstraintEngine
    source: dict = field(default_factory=dict)
    ratings: pd.DataFrame | None = None       # offline_user_id, item (snapshot index), rating


def active_real_version() -> str:
    return read_json(PROJECT_ROOT / "artifacts" / "real" / "active.json")["dataset_version"]


def load_real(cfg: dict, with_ratings: bool = False) -> Prepared:
    from pipelines.processed_io import ProcessedDataset

    settings = load_settings("real", environ={})
    version = cfg["data"].get("dataset_version") or active_real_version()
    ds = ProcessedDataset.for_version(settings, version)
    snap = snapshot_from_processed(ds)
    cat = catalog_from_settings(snap, PROJECT_ROOT, cfg["data"].get("supplement"))
    engine = build_engine(snap, ds.table("aliases"))
    source = {"kind": "real", "dataset_version": version, "processed_root": str(ds.root.relative_to(PROJECT_ROOT)),
              "processed_manifest": ds.manifest}
    ratings = ds.ratings(["offline_user_id", "item_idx", "rating"]) if with_ratings else None
    if ratings is not None:
        ratings = ratings.rename(columns={"item_idx": "item"})
    return Prepared(cat, engine, source, ratings)


# ---------------------------------------------------------------------------------------- fixture

FIXTURE_GENRES = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Mystery"]
FIXTURE_THEMES = ["School", "Space", "Music", "Military", "Detective"]
FIXTURE_STUDIOS = ["Studio Alpha", "Studio Beta", "Studio Gamma", "Studio Delta"]


def make_fixture(seed: int = 7, n_items: int = 80, n_users: int = 130) -> Prepared:
    """Tiny generated world (synthetic/fixture): users like items sharing their two favourite genres."""
    rng = np.random.default_rng(seed)
    rows, tag_rows = [], []
    for i in range(n_items):
        key = f"fixture:{seed}:{i + 1:03d}"
        g = rng.choice(len(FIXTURE_GENRES), size=rng.integers(1, 3), replace=False)
        score = None if i % 17 == 0 else round(float(rng.uniform(5.5, 9.2)), 2)
        eps = None if i % 23 == 0 else int(rng.choice([1, 12, 13, 24, 26, 50]))
        rows.append({"item_key": key, "title": f"Fixture Title {i + 1}", "title_english": None, "cover_url": None,
                     "score": score, "episodes": eps, "duration_minutes": 24.0,
                     "total_minutes": None if eps is None else eps * 24.0, "type": "TV",
                     "status": "Finished Airing", "members": int(rng.integers(100, 10000)), "aired_from": "2015",
                     "in_catalog": i % 31 != 30, "is_recommendable": score is not None,
                     "is_planner_eligible": score is not None, "is_explicit": i % 31 == 30})
        for c in g:
            tag_rows.append({"item_key": key, "namespace": "genre", "value": FIXTURE_GENRES[c]})
        if i % 3:
            tag_rows.append({"item_key": key, "namespace": "theme", "value": FIXTURE_THEMES[i % len(FIXTURE_THEMES)]})
        if i % 4:
            tag_rows.append({"item_key": key, "namespace": "studio", "value": FIXTURE_STUDIOS[i % len(FIXTURE_STUDIOS)]})
        if i % 2:
            tag_rows.append({"item_key": key, "namespace": "demographic", "value": ["Shounen", "Seinen"][i % 2]})
    catalog = pd.DataFrame(rows)
    tags = pd.DataFrame(tag_rows)
    snap = build_snapshot(catalog, tags, "fixture-v1", "fixture")
    cat = build_catalog(snap, None, None)
    engine = build_engine(snap, None)
    genre_of = [set(tags[(tags.item_key == k) & (tags.namespace == "genre")].value) for k in catalog.item_key]
    rat = []
    for u in range(n_users):
        fav = set(rng.choice(FIXTURE_GENRES, size=2, replace=False))
        size = int(rng.choice([1, 2, 3, 4, 7, 9, 12, 15]))
        liked = [i for i in range(n_items) if genre_of[i] & fav]
        rng.shuffle(liked)
        for i in liked[:size]:
            rat.append((u + 1, i, int(rng.integers(8, 11))))
        for i in rng.choice(n_items, size=5, replace=False):
            if i not in liked[:size]:
                rat.append((u + 1, int(i), int(rng.integers(3, 8))))
    ratings = pd.DataFrame(rat, columns=["offline_user_id", "item", "rating"]).drop_duplicates(
        ["offline_user_id", "item"])
    return Prepared(cat, engine, {"kind": "fixture", "label": "synthetic/fixture", "seed": seed,
                                  "dataset_version": "fixture-v1"}, ratings)


def load_prepared(cfg: dict, with_ratings: bool = False) -> Prepared:
    return make_fixture() if cfg["level"] == "fixture" else load_real(cfg, with_ratings)


# ---------------------------------------------------------------------------------------- prepare stage


def prepare(run, log) -> dict:
    cfg = run.config()
    prep = load_prepared(cfg, with_ratings=True)
    cat, r = prep.cat, prep.ratings
    thr = int(cfg["data"]["positive_threshold"])
    counts = {"rows_read": int(len(r))}
    file_checks = {}
    if prep.source["kind"] == "real":
        root = PROJECT_ROOT / prep.source["processed_root"]
        recorded = prep.source["processed_manifest"].get("files", {})
        for rel in ["catalog.parquet", "item_tags.parquet", "aliases.parquet",
                    *sorted(str(p.relative_to(root)) for p in (root / "ratings").glob("part-*.parquet"))]:
            actual = sha256_file(root / rel)
            file_checks[rel] = {"sha256": actual, "matches_manifest": recorded.get(rel) == actual}
        if not all(v["matches_manifest"] for v in file_checks.values()):
            raise SystemExit("processed files do not match their manifest checksums; refusing to prepare")
    valid_rating = r["rating"].between(1, 10)
    counts["invalid_rating"] = int((~valid_rating).sum())
    r = r[valid_rating]
    in_range = r["item"].between(0, cat.n - 1)
    counts["unknown_item"] = int((~in_range).sum())
    r = r[in_range]
    member = cat.member[r["item"].to_numpy()]
    counts["outside_catalogue"] = int((~member).sum())
    reasons = _outside_reasons(cat, r.loc[~member, "item"].to_numpy())
    r = r[member]
    dup = r.duplicated(["offline_user_id", "item"], keep=False)
    if dup.any():
        g = r[dup].groupby(["offline_user_id", "item"])["rating"].nunique()
        conflict = g[g > 1].index
        counts["conflicting_pairs_excluded"] = int(len(conflict))
        r = r.set_index(["offline_user_id", "item"])
        r = r[~r.index.isin(conflict)].reset_index()
        before = len(r)
        r = r.drop_duplicates(["offline_user_id", "item"])
        counts["identical_duplicates_removed"] = int(before - len(r))
    else:
        counts["conflicting_pairs_excluded"] = 0
        counts["identical_duplicates_removed"] = 0
    counts["rows_kept"] = int(len(r))
    pos = r[r["rating"] >= thr][["offline_user_id", "item"]].sort_values(["offline_user_id", "item"])
    pos = pos.reset_index(drop=True).astype({"offline_user_id": np.int64, "item": np.int32})
    sizes = pos.groupby("offline_user_id").size().rename("positive_size").reset_index()
    counts["positive_rows"] = int(len(pos))
    counts["users_with_positive"] = int(len(sizes))
    pos.to_parquet(run.work / "positives.parquet", index=False)
    sizes.to_parquet(run.work / "user_sizes.parquet", index=False)
    size_dist = {f"{lo}-{hi}": int(((sizes.positive_size >= lo) & (sizes.positive_size <= hi)).sum())
                 for lo, hi in ((1, 1), (2, 5), (6, 9), (10, 19), (20, 49), (50, 99), (100, 10**9))}
    manifest = {
        "prepare_version": PREPARE_VERSION, "created_at": now(), "level": cfg["level"],
        "source": {k: v for k, v in prep.source.items() if k != "processed_manifest"},
        "processed_dataset": ({"pipeline_version": prep.source["processed_manifest"].get("pipeline_version"),
                               "raw_source_sha256": prep.source["processed_manifest"].get("source_hashes"),
                               "config_fingerprint": prep.source["processed_manifest"].get("config_fingerprint")}
                              if prep.source["kind"] == "real" else None),
        "file_checks": file_checks,
        "catalogue": {"catalog_hash": cat.catalog_hash, "snapshot_rows": int(cat.n),
                      "members": int(len(cat.members)),
                      "membership_rule": "served (content rule: no Hentai / Erotica / Rx) and catalogue score known",
                      "field_coverage": cat.coverage(), "vocab_sizes": {ns: len(v) for ns, v in cat.vocab.items()},
                      "sources": cat.sources},
        "interactions": {**counts, "outside_catalogue_reasons": reasons},
        "labels": {"type": "proxy: high-rating preference set" if prep.source["kind"] == "real" else "synthetic/fixture",
                   "positive_rule": f"personal rating >= {thr} on a catalogue member",
                   "note": "no real collections exist; unrated / lower-rated titles are unobserved weak negatives"},
        "positive_set_size_distribution": size_dist,
        "id_mapping": {"items": "snapshot index <-> item_key (`mal:<MAL id>`); tie-break id = MAL id",
                       "users": "Kaggle offline user_id; never joined with application users"},
        "outputs": {"positives.parquet": sha256_df(pos), "user_sizes.parquet": sha256_df(sizes)},
    }
    write_json(run.path("data_manifest.json"), manifest)
    log(f"[prepare] members={len(cat.members)} kept={counts['rows_kept']:,} positives={len(pos):,} "
        f"users={len(sizes):,} catalog_hash={cat.catalog_hash[:12]}")
    return manifest


def _outside_reasons(cat: Alg2Catalog, items: np.ndarray) -> dict:
    snap = cat.snap
    out = {"not_served": 0, "explicit": 0, "no_catalogue_score": 0}
    if len(items) == 0:
        return out
    u, c = np.unique(items, return_counts=True)
    for i, n in zip(u.tolist(), c.tolist(), strict=True):
        if not snap.in_catalog[i]:
            out["not_served"] += n
        elif snap.explicit[i]:
            out["explicit"] += n
        else:
            out["no_catalogue_score"] += n
    return out


def load_positives(run) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(run.work / "positives.parquet"), pd.read_parquet(run.work / "user_sizes.parquet")


def ensure_prepared(run, log) -> None:
    if not (run.work / "positives.parquet").exists() or not run.exists("data_manifest.json"):
        prepare(run, log)


def run_root_rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
