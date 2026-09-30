"""Processed dataset layout shared by the synthetic and real pipelines.

data/processed/<mode>/<dataset_version>/
    catalog.parquet          one row per parsed title; row order = catalog_idx
    item_tags.parquet        item_key, namespace, value, source
    aliases.parquet          item_key, alias, kind
    relations.parquet        from_item, to_item, to_entity_type, to_external_id, to_name, relation_type, status, source_ref
    characters.parquet       source_id, name, item_key, role, source_ref
    scenes.parquet           verified / unverified scene records (optional)
    scene_images.parquet     frame manifest (optional)
    ratings/part-XXX.parquet offline_user_id:int64, item_idx:int32, rating:int8  (partitioned by user hash)
    labels/*.jsonl           derived label sets (e.g. weak MAL recommendation pairs)
    reports/*.json           preprocess / exclusion / quarantine reports
    manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.common import processed_root, read_json, sha256_file, write_json

RATING_SCHEMA = pa.schema([("offline_user_id", pa.int64()), ("item_idx", pa.int32()), ("rating", pa.int8())])

CATALOG_COLUMNS = [
    "item_key", "mal_id", "title", "title_english", "title_other", "synopsis", "score", "scored_by", "rank",
    "popularity", "members", "favorites", "type", "episodes", "duration_minutes", "total_minutes",
    "duration_raw", "duration_basis", "status", "content_rating", "source_material", "aired_raw", "aired_from",
    "aired_to", "premiered_season", "premiered_year", "cover_url", "provenance", "missing_fields", "in_catalog",
    "is_recommendable", "is_planner_eligible", "is_explicit", "exclusion_reasons",
]


def mix_hash(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 64-bit mix (splitmix64 finaliser) of integer ids -> uniform [0, 1)."""
    x = values.astype(np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class ProcessedWriter:
    def __init__(self, root: Path, n_partitions: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "ratings").mkdir(exist_ok=True)
        (self.root / "reports").mkdir(exist_ok=True)
        (self.root / "labels").mkdir(exist_ok=True)
        self.n_partitions = n_partitions
        self._writers: dict[int, pq.ParquetWriter] = {}

    # -------------------------------------------------------------- tables
    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def write_catalog(self, df: pd.DataFrame) -> Path:
        df = df.copy()
        for col in ("provenance", "missing_fields", "exclusion_reasons"):
            df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
        return self.write_table("catalog", df[CATALOG_COLUMNS])

    # -------------------------------------------------------------- ratings
    def append_ratings(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray) -> None:
        if len(users) == 0:
            return
        part = (mix_hash(users, 1234567) * self.n_partitions).astype(np.int64)
        for p in np.unique(part):
            sel = part == p
            tbl = pa.Table.from_arrays(
                [pa.array(users[sel].astype(np.int64)), pa.array(items[sel].astype(np.int32)),
                 pa.array(ratings[sel].astype(np.int8))], schema=RATING_SCHEMA)
            w = self._writers.get(int(p))
            if w is None:
                w = pq.ParquetWriter(self.root / "ratings" / f"part-{int(p):03d}.parquet", RATING_SCHEMA)
                self._writers[int(p)] = w
            w.write_table(tbl)

    def close_ratings(self) -> dict[str, int]:
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        return dedupe_partitions(self.root / "ratings")

    def write_report(self, name: str, obj: Any) -> None:
        write_json(self.root / "reports" / f"{name}.json", obj)

    def finalize(self, manifest: dict) -> dict:
        files = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".tmp"):
                files[str(p.relative_to(self.root))] = sha256_file(p)
        manifest = {**manifest, "files": files}
        write_json(self.root / "manifest.json", manifest)
        return manifest


def dedupe_partitions(ratings_dir: Path) -> dict[str, int]:
    """Drop duplicate (user, item) rows. No timestamps exist, so the first row
    in source-file order is kept (a reproducible rule, not "latest")."""
    stats = {"duplicate_rows_dropped": 0, "conflicting_pairs": 0, "rows": 0}
    for part in sorted(Path(ratings_dir).glob("part-*.parquet")):
        df = pq.read_table(part).to_pandas()
        dup_mask = df.duplicated(["offline_user_id", "item_idx"], keep="first")
        if dup_mask.any():
            grp = df.groupby(["offline_user_id", "item_idx"])["rating"].nunique()
            stats["conflicting_pairs"] += int((grp > 1).sum())
            stats["duplicate_rows_dropped"] += int(dup_mask.sum())
            df = df[~dup_mask]
            pq.write_table(pa.Table.from_pandas(df, schema=RATING_SCHEMA, preserve_index=False), part)
        stats["rows"] += len(df)
    return stats


# ------------------------------------------------------------------ readers


class ProcessedDataset:
    """Read access to one processed dataset version."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"processed dataset missing manifest: {self.root}")
        self.manifest = read_json(self.root / "manifest.json")

    @classmethod
    def for_version(cls, settings, version: str) -> ProcessedDataset:
        return cls(processed_root(settings, version))

    @property
    def version(self) -> str:
        return self.manifest["dataset_version"]

    def table(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def catalog(self) -> pd.DataFrame:
        df = self.table("catalog")
        for col in ("provenance", "missing_fields", "exclusion_reasons"):
            if col in df:
                df[col] = df[col].map(json.loads)
        return df

    def ratings(self, columns: list[str] | None = None) -> pd.DataFrame:
        parts = sorted((self.root / "ratings").glob("part-*.parquet"))
        if not parts:
            return pd.DataFrame(columns=["offline_user_id", "item_idx", "rating"])
        return pq.ParquetDataset([str(p) for p in parts]).read(columns=columns).to_pandas()

    def labels_path(self, name: str) -> Path:
        return self.root / "labels" / f"{name}.jsonl"
