"""Algorithm 2 item metadata, shared by offline experiments and the collection service.

Sources
  * catalogue snapshot (Kaggle MAL 2023 snapshot as processed by the pipeline): title, catalogue score,
    episodes, genres, studios, and the few themes / demographics fetched from Jikan;
  * an optional supplementary theme / demographic table keyed by MAL id (the team's projection in
    algorithm1_benchmark/data/catalog.jsonl.gz). Its values are unioned with the snapshot's own values.

Catalogue membership (the candidate universe of every scorer): served, not explicit, catalogue score
known. Other fields may be missing; nothing is imputed. A field is "known" for an item when the item has
at least one value in it: neither source distinguishes "known empty" from "missing", so the two are
reported together as missing.

`catalog_hash` covers every value a scorer can read, so models, encoders and feature caches are bound to
the exact metadata they were built with.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from app.core.text import norm
from app.recommendation.features import CatalogSnapshot

CATALOG_VERSION = "alg2-catalog-v1"
NAMESPACES = ("genre", "theme", "demographic", "studio")
SUPPLEMENT_NAMESPACES = ("theme", "demographic")
DEFAULT_SUPPLEMENT = "algorithm1_benchmark/data/catalog.jsonl.gz"


def order_id(key: str, fallback: int) -> int:
    """Numeric id used for tie-breaks: the MAL id for `mal:<id>`, the trailing number otherwise."""
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else fallback


def load_supplement(path: str | Path) -> dict[int, dict[str, list[str]]]:
    """MAL id -> {"theme": [...], "demographic": [...]} from a JSON-lines (optionally gzipped) table.

    Accepted fields per row: anime_id, themes (list), demographic (string or list) / demographics (list).
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[int, dict[str, list[str]]] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            demo = row.get("demographics", row.get("demographic"))
            demo = [demo] if isinstance(demo, str) else list(demo or [])
            themes = list(row.get("themes") or [])
            out[int(row["anime_id"])] = {"theme": [str(t) for t in themes if t], "demographic": [str(d) for d in demo if d]}
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Alg2Catalog:
    snap: CatalogSnapshot
    member: np.ndarray                     # bool per snapshot index: in the algorithm-2 catalogue
    members: np.ndarray                    # snapshot indices of members, sorted by anime_id
    anime_id: np.ndarray                   # int64 per snapshot index (tie-break order)
    score: np.ndarray
    episodes: np.ndarray
    vocab: dict[str, list[str]]
    X: dict[str, sp.csr_matrix]            # snapshot index x vocab, binary float32
    sizes: dict[str, np.ndarray]           # number of values per item and namespace
    sources: dict = field(default_factory=dict)
    catalog_hash: str = ""
    # genre + theme signature ids (deduplicates identical tag sets for max_tag_similarity)
    sig_id: np.ndarray = field(default=None)
    sig_X: dict[str, sp.csr_matrix] = field(default_factory=dict)
    sig_size: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.snap.n

    def known(self, ns: str) -> np.ndarray:
        return self.sizes[ns] > 0

    def tags_of(self, ns: str, i: int) -> list[str]:
        X = self.X[ns]
        return [self.vocab[ns][c] for c in X.indices[X.indptr[i]:X.indptr[i + 1]]]

    def coverage(self) -> dict:
        m = self.members
        out = {ns: round(float(self.known(ns)[m].mean()), 4) if len(m) else None for ns in NAMESPACES}
        out["episodes"] = round(float(np.isfinite(self.episodes[m]).mean()), 4) if len(m) else None
        out["score"] = round(float(np.isfinite(self.score[m]).mean()), 4) if len(m) else None
        return out


def build_catalog(snap: CatalogSnapshot, supplement: dict[int, dict[str, list[str]]] | None = None,
                  supplement_info: dict | None = None) -> Alg2Catalog:
    n = snap.n
    member = snap.in_catalog & ~snap.explicit & np.isfinite(snap.score)
    anime_id = np.array([order_id(k, i) for i, k in enumerate(snap.keys)], dtype=np.int64)
    members = np.flatnonzero(member)
    members = members[np.argsort(anime_id[members], kind="stable")]
    vocab: dict[str, list[str]] = {}
    X: dict[str, sp.csr_matrix] = {}
    sizes: dict[str, np.ndarray] = {}
    supp_used = {ns: 0 for ns in SUPPLEMENT_NAMESPACES}
    for ns in NAMESPACES:
        values: dict[int, set[str]] = {}
        canon: dict[str, str] = {}
        tm = snap.tags.get(ns)
        if tm is not None:
            for i in range(n):
                for v in tm.tags_of(i):
                    k = norm(v)
                    canon.setdefault(k, v)
                    values.setdefault(i, set()).add(k)
        if supplement and ns in SUPPLEMENT_NAMESPACES:
            for i, key in enumerate(snap.keys):
                if not key.startswith("mal:"):
                    continue
                row = supplement.get(int(key[4:]))
                if not row or not row.get(ns):
                    continue
                before = len(values.get(i, ()))
                for v in row[ns]:
                    k = norm(v)
                    canon.setdefault(k, v)
                    values.setdefault(i, set()).add(k)
                if member[i] and len(values[i]) > before:
                    supp_used[ns] += 1
        keys_sorted = sorted(canon, key=lambda k: (canon[k], k))
        col = {k: c for c, k in enumerate(keys_sorted)}
        rows, cols = [], []
        for i, vs in values.items():
            for k in vs:
                rows.append(i)
                cols.append(col[k])
        mat = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (np.array(rows, dtype=np.int64),
                                                                   np.array(cols, dtype=np.int64))),
                            shape=(n, max(1, len(keys_sorted))))
        mat.sum_duplicates()
        mat.data[:] = 1.0
        mat.sort_indices()
        vocab[ns] = [canon[k] for k in keys_sorted]
        X[ns] = mat
        sizes[ns] = np.asarray(mat.sum(axis=1)).ravel().astype(np.int32)
    cat = Alg2Catalog(snap=snap, member=member, members=members, anime_id=anime_id,
                      score=snap.score.astype(np.float64), episodes=snap.episodes.astype(np.float64),
                      vocab=vocab, X=X, sizes=sizes)
    cat.sources = {"catalog_version": CATALOG_VERSION, "dataset_version": snap.dataset_version,
                   "dataset_mode": snap.dataset_mode, "members": int(len(members)),
                   "supplement": supplement_info, "supplement_values_added": supp_used}
    _signatures(cat)
    cat.catalog_hash = catalog_hash(cat)
    return cat


def _signatures(cat: Alg2Catalog) -> None:
    """Group items by identical (genre set, theme set) so pairwise Jaccard runs on unique signatures."""
    g, t = cat.X["genre"], cat.X["theme"]
    sig_of: dict[tuple, int] = {}
    sig_id = np.zeros(cat.n, dtype=np.int64)
    reps: list[int] = []
    for i in range(cat.n):
        key = (tuple(g.indices[g.indptr[i]:g.indptr[i + 1]]), tuple(t.indices[t.indptr[i]:t.indptr[i + 1]]))
        s = sig_of.get(key)
        if s is None:
            s = sig_of[key] = len(reps)
            reps.append(i)
        sig_id[i] = s
    reps_arr = np.array(reps, dtype=np.int64)
    cat.sig_id = sig_id
    for ns in ("genre", "theme"):
        cat.sig_X[ns] = cat.X[ns][reps_arr]
        cat.sig_size[ns] = cat.sizes[ns][reps_arr].astype(np.float32)


def catalog_hash(cat: Alg2Catalog) -> str:
    h = hashlib.sha256(CATALOG_VERSION.encode())
    m = cat.members
    h.update(np.ascontiguousarray(cat.anime_id[m]).tobytes())
    h.update(np.ascontiguousarray(np.round(np.nan_to_num(cat.score[m], nan=-1.0), 4)).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(cat.episodes[m], nan=-1.0)).tobytes())
    for ns in NAMESPACES:
        h.update(ns.encode())
        h.update(json.dumps(cat.vocab[ns], ensure_ascii=False).encode())
        sub = cat.X[ns][m]
        h.update(np.ascontiguousarray(sub.indptr).tobytes())
        h.update(np.ascontiguousarray(sub.indices).tobytes())
    return h.hexdigest()


def catalog_from_settings(snap: CatalogSnapshot, project_root: Path, supplement_path: str | None) -> Alg2Catalog:
    """Build the catalogue with the configured supplement (None / missing file -> snapshot values only)."""
    if supplement_path and snap.dataset_mode == "real":
        p = Path(supplement_path)
        p = p if p.is_absolute() else project_root / p
        if p.exists():
            info = {"path": str(supplement_path), "sha256": file_sha256(p), "namespaces": list(SUPPLEMENT_NAMESPACES),
                    "rule": "union with snapshot values; MAL id join"}
            return build_catalog(snap, load_supplement(p), info)
        return build_catalog(snap, None, {"path": str(supplement_path), "status": "missing"})
    return build_catalog(snap, None, None)
