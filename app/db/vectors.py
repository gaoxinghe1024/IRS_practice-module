"""Embedding vectors stored one row per vector in the business database.

A vector is the raw bytes of one matrix row in the dtype / dim recorded in `vector_sets`, together with the
SHA-256 of the whole matrix in load order. Readers rebuild the matrix and check that checksum, so a partial or
edited set is refused instead of being served next to a different id mapping.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import models as M


class VectorSetMismatch(RuntimeError):
    pass


def matrix_sha256(mat: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mat).tobytes()).hexdigest()


def row_blobs(mat: np.ndarray) -> list[bytes]:
    mat = np.ascontiguousarray(mat)
    return [row.tobytes() for row in mat]


def record_set(conn, name: str, version: str, mat: np.ndarray, info: dict | None = None) -> dict:
    row = {"name": name, "dataset_version": version, "dim": int(mat.shape[1]), "dtype": str(mat.dtype),
           "count": int(mat.shape[0]), "sha256": matrix_sha256(mat), "info": info or {}}
    stmt = sqlite_insert(M.VectorSet.__table__).values(row)
    conn.execute(stmt.on_conflict_do_update(index_elements=["name"], set_={k: v for k, v in row.items() if k != "name"}))
    return row


def drop_set(conn, name: str) -> None:
    conn.execute(delete(M.VectorSet).where(M.VectorSet.name == name))


def read_set(conn, name: str) -> dict | None:
    row = conn.execute(select(M.VectorSet.__table__).where(M.VectorSet.name == name)).mappings().first()
    return dict(row) if row is not None else None


def assemble(meta: dict, blobs: list[bytes], version: str) -> np.ndarray:
    """Matrix from row blobs in load order, checked against the set's version, count and checksum."""
    if meta["dataset_version"] != version:
        raise VectorSetMismatch(f"vector set {meta['name']} belongs to {meta['dataset_version']}, not {version}")
    if len(blobs) != meta["count"]:
        raise VectorSetMismatch(f"vector set {meta['name']}: {len(blobs)} rows, expected {meta['count']}")
    buf = bytearray().join(blobs)                   # writable buffer, one copy
    mat = np.frombuffer(buf, dtype=np.dtype(meta["dtype"])).reshape(len(blobs), meta["dim"]) if blobs else \
        np.zeros((0, meta["dim"]), dtype=np.dtype(meta["dtype"]))
    if matrix_sha256(mat) != meta["sha256"]:
        raise VectorSetMismatch(f"vector set {meta['name']}: checksum mismatch")
    return mat
