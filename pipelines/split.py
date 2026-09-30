"""Shared split manifests for the two experiment protocols.

ranking   eligible users (>= min ratings and >= 2 ratings >= positive_threshold) hold out one
          high-rated validation positive and one independent high-rated test positive; everything
          else is train. Used for Hits/MRR/NDCG and for the served models.
explicit  per-user random 80/10/10 (users with >= min ratings); used only for MAE/RMSE diagnostics.

No timestamps exist in the snapshot, so neither protocol is temporal. Random
choices use a seeded hash of (user, item), so a split is reproducible and has a
fingerprint recorded in its manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings
from pipelines.common import latest_dataset_version, sha256_json, utc_now_iso, write_json
from pipelines.processed_io import ProcessedDataset, mix_hash


def _fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    parts = {}
    for name, df in frames.items():
        if name == "train":
            continue
        pairs = np.sort(df.offline_user_id.to_numpy().astype(np.int64) * 10_000_000 + df.item_idx.to_numpy())
        parts[name] = sha256_json(pairs[:: max(1, len(pairs) // 200_000)].tolist() + [len(pairs)])
    return sha256_json(parts)[:16]


def _count_buckets(n_per_user: pd.Series) -> dict[str, int]:
    bins = [0, 1, 3, 8, 20, 50, 100, 250, 10**9]
    labels = ["1", "2-3", "4-8", "9-20", "21-50", "51-100", "101-250", ">250"]
    cats = pd.cut(n_per_user, bins=bins, labels=labels, right=True)
    return {str(k): int(v) for k, v in cats.value_counts().sort_index().items()}


def make_splits(settings: Settings, version: str | None = None, log=print) -> dict[str, Any]:
    version = version or latest_dataset_version(settings)
    ds = ProcessedDataset.for_version(settings, version)
    pipe = settings.pipeline
    seed = int(pipe.get("split_seed", 20260930))
    thr = int(pipe.get("positive_threshold", 8))
    min_rank = int(pipe.get("min_ratings_ranking_eval", 5))
    min_exp = int(pipe.get("min_ratings_explicit_eval", 5))
    fracs = pipe.get("explicit_split", [0.8, 0.1, 0.1])

    df = ds.ratings()
    df = df.sort_values(["offline_user_id", "item_idx"], kind="stable").reset_index(drop=True)
    u = df.offline_user_id.to_numpy()
    key = mix_hash(u.astype(np.int64) * 1_000_003 + df.item_idx.to_numpy().astype(np.int64), seed)
    df["_k"] = key
    n_user = df.groupby("offline_user_id")["rating"].transform("size")
    is_pos = df.rating.to_numpy() >= thr
    n_pos = pd.Series(is_pos, index=df.index).groupby(df.offline_user_id).transform("sum")

    out_root = ds.root / "splits"
    manifests = {}

    # ------------------------------------------------------------------ ranking
    eligible = (n_user >= min_rank) & (n_pos >= 2)
    df["_pos_rank"] = -1
    pos_rows = df[eligible & is_pos].sort_values(["offline_user_id", "_k"], kind="stable")
    ranks = pos_rows.groupby("offline_user_id").cumcount()
    df.loc[pos_rows.index, "_pos_rank"] = ranks.to_numpy()
    part = np.where(df["_pos_rank"].to_numpy() == 0, "test", np.where(df["_pos_rank"].to_numpy() == 1, "val", "train"))
    frames = {name: df.loc[part == name, ["offline_user_id", "item_idx", "rating"]].reset_index(drop=True)
              for name in ("train", "val", "test")}
    rdir = out_root / "ranking"
    rdir.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_parquet(rdir / f"{name}.parquet", index=False)
    users_all = df.groupby("offline_user_id").size()
    elig_users = df.loc[eligible, "offline_user_id"].nunique()
    m_rank = {
        "protocol": "ranking_leave_two_high_out", "dataset_version": version, "seed": seed,
        "positive_threshold": thr, "min_ratings": min_rank, "created_at": utc_now_iso(),
        "counts": {k: int(len(v)) for k, v in frames.items()},
        "users": {"total": int(len(users_all)), "eval_eligible": int(elig_users),
                  "train_only": int(len(users_all) - elig_users)},
        "user_rating_count_buckets": _count_buckets(users_all),
        "fingerprint": _fingerprint(frames),
        "note": "validation and test positives are distinct high ratings of the same user; no timestamps exist",
    }
    write_json(rdir / "manifest.json", m_rank)
    manifests["ranking"] = m_rank
    log(f"[split] ranking: {m_rank['counts']} eligible_users={elig_users}")

    # ------------------------------------------------------------------ explicit
    exp_elig = n_user >= min_exp
    order = df[exp_elig].sort_values(["offline_user_id", "_k"], kind="stable")
    pos_in_user = order.groupby("offline_user_id").cumcount().to_numpy()
    size = n_user[order.index].to_numpy()
    frac = pos_in_user / size
    lab = np.where(frac < fracs[0], "train", np.where(frac < fracs[0] + fracs[1], "val", "test"))
    part2 = pd.Series("train", index=df.index, dtype=object)
    part2.loc[order.index] = lab
    frames2 = {name: df.loc[part2.to_numpy() == name, ["offline_user_id", "item_idx", "rating"]].reset_index(drop=True)
               for name in ("train", "val", "test")}
    edir = out_root / "explicit"
    edir.mkdir(parents=True, exist_ok=True)
    for name, f in frames2.items():
        f.to_parquet(edir / f"{name}.parquet", index=False)
    m_exp = {
        "protocol": "explicit_random_per_user", "dataset_version": version, "seed": seed,
        "fractions": fracs, "min_ratings": min_exp, "created_at": utc_now_iso(),
        "counts": {k: int(len(v)) for k, v in frames2.items()},
        "users": {"eval_eligible": int(df.loc[exp_elig, "offline_user_id"].nunique())},
        "fingerprint": _fingerprint(frames2),
    }
    write_json(edir / "manifest.json", m_exp)
    manifests["explicit"] = m_exp
    log(f"[split] explicit: {m_exp['counts']}")
    return manifests


def load_split(settings: Settings, protocol: str, part: str, version: str | None = None) -> pd.DataFrame:
    version = version or latest_dataset_version(settings)
    return pd.read_parquet(settings.processed_dir / version / "splits" / protocol / f"{part}.parquet")


def split_manifest(settings: Settings, protocol: str, version: str | None = None) -> dict:
    import json

    version = version or latest_dataset_version(settings)
    return json.loads((settings.processed_dir / version / "splits" / protocol / "manifest.json").read_text())


def split_dir(settings: Settings, protocol: str, version: str) -> Path:
    return settings.processed_dir / version / "splits" / protocol
