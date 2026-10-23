# -*- coding: utf-8 -*-
"""
data_io.py — loaders for the AnimeFinder 20260918_A_final_v1 data package.

Everything is read read-only from <package>/data. Nothing is written back.

ID NOTE (verified against the package, do not "optimise" away):
  anime_index != anime_id - 1  and  user_index != user_id - 1.
  The CSR matrices are indexed by anime_index / user_index; the jsonl / csv
  side files carry the original MAL ids. All joins MUST go through
  cf/anime_mapping.csv and cf/user_mapping.csv.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config as C


def _log(msg):
    print(f"[data_io] {msg}", flush=True)


def load_ratings_train():
    """19,387,811 training ratings as CSR (users x items), raw 1-10 scores."""
    t = time.time()
    z = np.load(C.F_TRAIN_NPZ)
    R = sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int32),
                       z["indptr"].astype(np.int32)), shape=tuple(int(v) for v in z["shape"]))
    R.sort_indices()
    _log(f"ratings_train {R.shape} nnz={R.nnz:,} loaded in {time.time()-t:.1f}s")
    return R


def load_observed_train():
    """Boolean observation mask: separates 'rated 0 after centring' from 'not rated'."""
    z = np.load(C.F_OBSERVED_NPZ)
    M = sp.csr_matrix((z["data"].astype(bool), z["indices"].astype(np.int32),
                       z["indptr"].astype(np.int32)), shape=tuple(int(v) for v in z["shape"]))
    M.sort_indices()
    return M


def load_validation():
    """2,283,673 held-out ratings (already excluded from ratings_train)."""
    t = time.time()
    df = pd.read_csv(C.F_VALIDATION,
                     dtype={"source": "string", "user_id": np.int64,
                            "anime_id": np.int64, "rating": np.float32,
                            "source_record_no": np.int64})
    _log(f"ratings_validation {len(df):,} rows loaded in {time.time()-t:.1f}s")
    return df


def load_mappings():
    """Returns (anime_id -> anime_index, user_id -> user_index) dicts."""
    am = pd.read_csv(C.F_ANIME_MAP)
    um = pd.read_csv(C.F_USER_MAP)
    a2i = dict(zip(am["anime_id"].to_numpy(), am["anime_index"].to_numpy()))
    u2i = dict(zip(um["user_id"].to_numpy(), um["user_index"].to_numpy()))
    _log(f"mappings: {len(a2i):,} anime, {len(u2i):,} users "
         f"(id != index+1, joins must use these maps)")
    return a2i, u2i


def load_item_features(a2i, n_items):
    """GA item features: genres / themes / novelty_train / quality_train / duration."""
    t = time.time()
    genres = [()] * n_items
    themes = [()] * n_items
    qual = np.zeros(n_items, dtype=np.float32)
    nov = np.zeros(n_items, dtype=np.float32)
    dur = np.zeros(n_items, dtype=np.float32)
    seen = np.zeros(n_items, dtype=bool)
    with open(C.F_ITEM_FEATURES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            i = a2i.get(int(d["anime_id"]))
            if i is None:
                continue
            genres[i] = tuple(d.get("genres") or ())
            themes[i] = tuple(d.get("themes") or ())
            qual[i] = float(d.get("quality_train") or 0.0)
            nov[i] = float(d.get("novelty_train") or 0.0)
            dur[i] = float(d.get("total_duration_minutes") or 0.0)
            seen[i] = True
    _log(f"item_features matched {seen.sum():,}/{n_items:,} items in {time.time()-t:.1f}s")
    return dict(genres=genres, themes=themes, quality=qual, novelty=nov,
                duration=dur, seen=seen)


def load_release_years(a2i, n_items):
    """aired_start -> release year, used by the GA 'fresh ratio' term."""
    t = time.time()
    year = np.zeros(n_items, dtype=np.int32)
    with open(C.F_ANIME_RUNTIME, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            i = a2i.get(int(d.get("anime_id") or 0))
            if i is None:
                continue
            s = str(d.get("aired_start") or "")
            if len(s) >= 4 and s[:4].isdigit():
                year[i] = int(s[:4])
    nz = year[year > 0]
    _log(f"release years matched {len(nz):,}/{n_items:,} items in {time.time()-t:.1f}s "
         f"(min={nz.min() if len(nz) else 0} max={year.max()})")
    return year


def load_user_genre_weights(genre_index, u2i):
    """Per-user genre preference weights (train split only).

    Used by the GA 'personalization' term when GA_FITNESS_MODE == 'handoff'.
    Returns dict user_index -> float32 vector over the global genre vocabulary.
    """
    t = time.time()
    out = {}
    with open(os.path.join(C.CF_DIR, "train_statistics.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("entity_type") != "user":
                continue
            gw = d.get("genre_weights")
            if not gw:
                continue
            ui = u2i.get(int(d["entity_id"]))
            if ui is None:
                continue
            v = np.zeros(len(genre_index), dtype=np.float32)
            for g, w in gw.items():
                j = genre_index.get(g)
                if j is not None:
                    v[j] = w
            out[ui] = v
    _log(f"user genre weights loaded for {len(out):,} users in {time.time()-t:.1f}s")
    return out


def build_validation_indexed(val_df, a2i, u2i):
    """Attach matrix indices to the held-out table (drops rows that cannot be mapped)."""
    u = val_df["user_id"].map(u2i).to_numpy()
    i = val_df["anime_id"].map(a2i).to_numpy()
    keep = pd.notna(u) & pd.notna(i)
    v = val_df.loc[keep, ["rating"]].copy()
    v["u"] = u[keep].astype(np.int64)
    v["i"] = i[keep].astype(np.int64)
    v["rating"] = val_df.loc[keep, "rating"].to_numpy()
    _log(f"validation mapped to matrix indices: {len(v):,}/{len(val_df):,} rows")
    return v.reset_index(drop=True)


def build_eligible_cohort(R, v):
    """Users eligible for the leave-one-out protocol:
       - >= MIN_TRAIN_RATINGS training ratings (the ICF seed rule), and
       - >= 1 held-out record with rating >= HIGH_RATING.
    Returns the hidden record per user (deterministic tie-break).
    """
    cnt = np.diff(R.indptr)
    high = v[v["rating"] >= C.HIGH_RATING]
    high = high.sort_values(["u", "rating", "i"], ascending=[True, False, True])
    hidden = high.drop_duplicates("u", keep="first").copy()
    keep = cnt[hidden["u"].to_numpy()] >= C.MIN_TRAIN_RATINGS
    hidden = hidden[keep].reset_index(drop=True)
    _log(f"eligible users after filters: {len(hidden):,} "
         f"(>= {C.MIN_TRAIN_RATINGS} train ratings and a >= {C.HIGH_RATING} held-out record)")
    return hidden


def sample_cohort(hidden, size=C.COHORT_SIZE, seed=C.RANDOM_SEED):
    """Fixed random sample so every configuration is scored on the same users."""
    if size >= len(hidden):
        return hidden.reset_index(drop=True)
    pick = np.random.default_rng(seed).choice(len(hidden), size=size, replace=False)
    pick.sort()
    return hidden.iloc[pick].reset_index(drop=True)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_log(path, msg):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
