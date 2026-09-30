"""In-memory catalogue snapshot used by every algorithm (serving and offline experiments).

The snapshot is built from the same two tables in both contexts (catalogue rows
+ item tags), either read from the processed Parquet files or from the business
database, so online and offline code paths share one representation.
Unknown values stay NaN / empty sets; nothing is imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from app.core.text import norm

SIM_NAMESPACES = ("genre", "theme", "demographic", "studio")
PROFILE_NAMESPACES = ("genre", "theme", "demographic", "studio")


@dataclass
class TagMatrix:
    namespace: str
    vocab: list[str]
    col: dict[str, int]
    X: sp.csr_matrix            # items x vocab, binary float32
    sizes: np.ndarray           # tags per item

    def tags_of(self, i: int) -> list[str]:
        row = self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]
        return [self.vocab[c] for c in row]

    def cols_of(self, i: int) -> np.ndarray:
        return self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]


@dataclass
class CatalogSnapshot:
    dataset_version: str
    dataset_mode: str
    keys: np.ndarray
    index: dict[str, int]
    titles: list[str]
    score: np.ndarray
    episodes: np.ndarray
    duration: np.ndarray        # minutes per episode
    total_minutes: np.ndarray
    type: np.ndarray
    status: np.ndarray
    members: np.ndarray
    year: np.ndarray
    in_catalog: np.ndarray
    recommendable: np.ndarray
    planner_eligible: np.ndarray
    explicit: np.ndarray
    tags: dict[str, TagMatrix]
    covers: list = field(default_factory=list)
    titles_english: list = field(default_factory=list)
    train_interactions: np.ndarray = field(default=None)  # filled from the active artifact

    @property
    def n(self) -> int:
        return len(self.keys)

    def idx(self, key: str) -> int | None:
        return self.index.get(key)

    def tag_names(self, i: int, namespaces=("genre", "theme")) -> list[str]:
        out: list[str] = []
        for ns in namespaces:
            if ns in self.tags:
                out.extend(self.tags[ns].tags_of(i))
        return out

    def with_train_interactions(self, counts: np.ndarray | None) -> CatalogSnapshot:
        self.train_interactions = (np.zeros(self.n, dtype=np.int64) if counts is None
                                   else np.asarray(counts, dtype=np.int64))
        return self


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def build_snapshot(catalog: pd.DataFrame, tags: pd.DataFrame, dataset_version: str, dataset_mode: str,
                   namespaces=("genre", "theme", "demographic", "studio", "source")) -> CatalogSnapshot:
    catalog = catalog.reset_index(drop=True)
    keys = catalog["item_key"].astype(str).to_numpy()
    index = {k: i for i, k in enumerate(keys)}
    year = pd.to_numeric(catalog["aired_from"].astype("string").str.slice(0, 4), errors="coerce").to_numpy(
        dtype=np.float64)
    tag_mats: dict[str, TagMatrix] = {}
    tags = tags if len(tags) else pd.DataFrame(columns=["item_key", "namespace", "value"])
    for ns in namespaces:
        sub = tags[tags["namespace"] == ns]
        vocab_raw = sorted(set(sub["value"].astype(str)), key=lambda v: (norm(v), v))
        # merge values that differ only by case / width
        vocab: list[str] = []
        col: dict[str, int] = {}
        for v in vocab_raw:
            k = norm(v)
            if k not in col:
                col[k] = len(vocab)
                vocab.append(v)
        rows = sub["item_key"].map(index)
        ok = rows.notna().to_numpy()
        r = rows[ok].astype(int).to_numpy()
        c = sub["value"][ok].map(lambda v, col=col: col[norm(str(v))]).to_numpy()
        X = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(keys), max(1, len(vocab))))
        X.sum_duplicates()
        X.data[:] = 1.0
        tag_mats[ns] = TagMatrix(ns, vocab, col, X, np.asarray(X.sum(axis=1)).ravel())

    def _bool(col: str) -> np.ndarray:
        return catalog[col].fillna(False).astype(bool).to_numpy()

    return CatalogSnapshot(
        dataset_version=dataset_version, dataset_mode=dataset_mode, keys=keys, index=index,
        titles=catalog["title"].astype(str).tolist(), score=_num(catalog["score"]),
        episodes=_num(catalog["episodes"]), duration=_num(catalog["duration_minutes"]),
        total_minutes=_num(catalog["total_minutes"]),
        type=catalog["type"].astype(object).where(catalog["type"].notna(), None).to_numpy(),
        status=catalog["status"].astype(object).where(catalog["status"].notna(), None).to_numpy(),
        members=_num(catalog["members"]), year=year, in_catalog=_bool("in_catalog"),
        recommendable=_bool("is_recommendable"), planner_eligible=_bool("is_planner_eligible"),
        explicit=_bool("is_explicit"), tags=tag_mats,
        covers=[c if isinstance(c, str) and c else None for c in catalog["cover_url"].tolist()],
        titles_english=[t if isinstance(t, str) and t else None for t in catalog["title_english"].tolist()],
    ).with_train_interactions(None)


def snapshot_from_processed(ds) -> CatalogSnapshot:
    return build_snapshot(ds.catalog(), ds.table("item_tags"), ds.version, ds.manifest["dataset_mode"])


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    out = np.zeros_like(x)
    if hi - lo < 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def stable_order(scores: np.ndarray, keys: np.ndarray, secondary: np.ndarray | None = None) -> np.ndarray:
    """Descending by score, then by secondary (desc, e.g. coverage), then item_key ascending."""
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    if secondary is None:
        return np.lexsort((keys, -s))
    sec = np.nan_to_num(np.asarray(secondary, dtype=np.float64), nan=-np.inf)
    return np.lexsort((keys, -sec, -s))
