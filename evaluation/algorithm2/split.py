"""`split` stage: user selection, user-level partitions, query manifest, sampled pairs, selection protocol.

Users (proxy collections = positive preference sets)
  main    positive set >= main_min_positive (visible history >= 5 after holding out the target)
  sparse  positive set within sparse_range (history 1-4), reported separately
Selection: eligible users are ordered by a seeded hash of their id and the first N are kept (main and
sparse separately). Partitions: within each positive-set-size bucket, users are ordered by a second
hash and cut 70 / 15 / 15, so every user (and all of their records) lives in exactly one partition.
The smoke level is a nested subset of the main split (first users of each group/partition by a third
hash), so a user never changes partition between levels.

Queries (order of operations per query): choose the target t -> remove it, H = positives - {t} ->
everything else reads H only. Validation / test: one target per collection, evaluated under rules
R0-R3. Training: up to `train_targets_per_collection` targets per collection, rule R0 only.
R1's keyword is H's most frequent genre (ties: normalised name ascending), never the target's.

Pairs (shared by B0 / B1 / C0): per training / validation query, the positive t and up to
`per_query` weak negatives drawn uniformly (seeded per query) from the R0 candidates (catalogue minus
H), excluding every positive member of the collection. Validation pairs only serve early stopping.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from app.core.text import norm
from evaluation.algorithm2.common import now, sha256_df, write_json
from evaluation.algorithm2.data import ensure_prepared, load_positives, load_prepared

PARTITIONS = ("train", "val", "test")


def _hash01(ids: np.ndarray, seed: int) -> np.ndarray:
    from pipelines.processed_io import mix_hash

    return mix_hash(np.asarray(ids, dtype=np.int64), int(seed))


def _qhash(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def select_users(sizes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sc = cfg["split"]
    seed = int(sc["split_seed"])
    lo, hi = sc["sparse_range"]
    s = sizes.copy()
    s["group"] = np.where(s.positive_size >= sc["main_min_positive"], "main",
                          np.where(s.positive_size.between(lo, hi), "sparse", "excluded"))
    out = []
    for group, n_take in (("main", sc["main_users"]), ("sparse", sc["sparse_users"])):
        g = s[s.group == group].copy()
        g["h_select"] = _hash01(g.offline_user_id.to_numpy(), seed)
        g = g.sort_values(["h_select", "offline_user_id"]).head(int(n_take))
        out.append(g)
    users = pd.concat(out, ignore_index=True)
    edges = list(sc["size_buckets"])
    users["bucket"] = np.where(users.group == "sparse", f"{lo}-{hi}",
                               pd.cut(users.positive_size, bins=edges + [np.inf], right=False,
                                      labels=[f"{a}-{b - 1}" if np.isfinite(b) else f"{a}+"
                                              for a, b in zip(edges, edges[1:] + [np.inf], strict=True)]
                                      ).astype(str))
    users["h_part"] = _hash01(users.offline_user_id.to_numpy(), seed + 1)
    users["partition"] = ""
    fr = sc["fractions"]
    for _, idx in users.groupby(["group", "bucket"]).groups.items():
        sub = users.loc[idx].sort_values(["h_part", "offline_user_id"])
        n = len(sub)
        n_tr = int(round(n * fr[0]))
        n_va = int(round(n * fr[1]))
        labels = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        users.loc[sub.index, "partition"] = labels
    users["h_smoke"] = _hash01(users.offline_user_id.to_numpy(), seed + 2)
    users["in_smoke"] = False
    for group, n_smoke in (("main", sc["smoke_main_users"]), ("sparse", sc["smoke_sparse_users"])):
        for p, f in zip(PARTITIONS, fr, strict=True):
            sub = users[(users.group == group) & (users.partition == p)].sort_values(["h_smoke", "offline_user_id"])
            users.loc[sub.index[:int(round(n_smoke * f))], "in_smoke"] = True
    users = users.sort_values("offline_user_id").reset_index(drop=True)
    return users[["offline_user_id", "group", "positive_size", "bucket", "partition", "in_smoke"]]


def top_genre_keyword(cat, history: np.ndarray) -> str | None:
    if len(history) == 0:
        return None
    counts = np.asarray(cat.X["genre"][history].sum(axis=0)).ravel()
    if not counts.any():
        return None
    best = counts.max()
    names = sorted(norm(cat.vocab["genre"][c]) for c in np.flatnonzero(counts == best))
    return names[0]


def build_queries(cat, users: pd.DataFrame, positives: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sc = cfg["split"]
    seed = int(sc["split_seed"])
    n_train = int(sc["train_targets_per_collection"])
    pos_by_user = positives[positives.offline_user_id.isin(users.offline_user_id)].groupby("offline_user_id")["item"]
    pos_map = {u: np.asarray(v.to_numpy(), dtype=np.int64) for u, v in pos_by_user}
    rows = []
    for u, group, part in users[["offline_user_id", "group", "partition"]].itertuples(index=False):
        items = np.sort(pos_map[u])
        order = np.argsort(_hash01(items * 1_000_003 + u % 1_000_003, seed + 3), kind="stable")
        targets = items[order[: (n_train if part == "train" else 1)]]
        for t in targets:
            vis = items[items != t]
            rows.append({"query_id": f"{part}-{u}-{cat.snap.keys[t]}", "offline_user_id": int(u), "group": group,
                         "partition": part, "collection_id": f"proxy:{u}", "target": int(t),
                         "target_key": str(cat.snap.keys[t]), "target_anime_id": int(cat.anime_id[t]),
                         "visible": vis.astype(np.int64), "history_size": int(len(vis)),
                         "r1_keyword": top_genre_keyword(cat, vis)})
    q = pd.DataFrame(rows)
    return q.sort_values(["partition", "offline_user_id", "target"]).reset_index(drop=True)


def sample_pairs(cat, queries: pd.DataFrame, positives: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Positive + weak negatives for train and validation queries (rule R0)."""
    nc = cfg["negatives"]
    neg_seed, per_q = int(nc["negative_seed"]), int(nc["per_query"])
    pos_sets = {u: set(v.tolist()) for u, v in
                positives[positives.offline_user_id.isin(queries.offline_user_id)].groupby("offline_user_id")["item"]}
    members = cat.members
    rows = []
    shortfall = 0
    for q in queries[queries.partition.isin(["train", "val"])].itertuples(index=False):
        rows.append((q.query_id, q.partition, int(q.target), 1))
        excluded = pos_sets[q.offline_user_id]                      # all known positives, incl. t and H
        rng = np.random.default_rng([neg_seed, _qhash(q.query_id)])
        picked: list[int] = []
        tries = 0
        while len(picked) < per_q and tries < 50:
            draw = members[rng.integers(0, len(members), size=per_q * 2)]
            for i in draw.tolist():
                if i not in excluded and i not in picked:
                    picked.append(i)
                    if len(picked) == per_q:
                        break
            tries += 1
        shortfall += per_q - len(picked)
        rows.extend((q.query_id, q.partition, int(i), 0) for i in picked)
    pairs = pd.DataFrame(rows, columns=["query_id", "partition", "item", "label"])
    pairs.attrs["shortfall"] = shortfall
    return pairs


def selection_protocol(cfg: dict) -> dict:
    sel = cfg["selection"]
    return {
        "written_at": now(),
        "status": "pre-registered before any model training or test evaluation",
        "primary_metric": "user-macro HR@8 on rule R0, main group (visible history >= 5), queries with more than 8 "
                          "candidates, full-catalogue candidate ranking",
        "validation_use": "A1 weight, GBDT / two-tower hyper-parameters, early stopping and the pre-selection of the "
                          "candidate to deploy use validation users only; test users are evaluated once, after freeze",
        "tie_break": {"margin": sel["tie_margin"], "order": ["higher NDCG@8", "lower P95 scoring latency (same machine)",
                                                             "lower complexity (A1 < B0 < B1 < C0; fewer trees x leaves)"]},
        "a1": {"grid": cfg["a1_weights"], "rule": "highest validation HR@8; within the margin: higher NDCG@8, then the "
                                                  "weight closest to 0.15, then the smaller weight"},
        "trial_selection": "per model family: highest validation HR@8 (training seed 42); within the margin the tie-break "
                           "order above",
        "pre_selection": "the candidate to deploy is the family with the highest validation HR@8 among A1, B0, B1, C0 "
                         "(learned families: mean over training seeds); within the margin the tie-break order above",
        "conservative_fallback": "A1 (validation-tuned formula; identical to A0 when the chosen weight is 0.15)",
        "deployed_artifact": f"training seed {cfg['deploy_seed']} of the pre-selected family; its own test metrics "
                             "are reported next to the seed mean and decide the confirmation",
        "seeds": cfg["seeds"],
        "confirmation_on_test": {
            "vs_A0": [f"user-macro AUC >= {sel['auc_min']}", f"HR@8 - A0 HR@8 >= {sel['hr_gain_min']}",
                      "strict rule violation rate = 0",
                      "paired user bootstrap 95% CI of (HR@8 - A0 HR@8) has lower bound > 0"],
            "vs_A1": "the learned candidate also needs a paired 95% CI lower bound > 0 for (HR@8 - A1 HR@8); otherwise "
                     "the extra complexity is not justified and A1 is kept",
            "bootstrap": {"unit": "user", "replicates": sel["bootstrap"], "seed": sel["bootstrap_seed"]},
        },
        "if_not_confirmed": "deploy the conservative fallback A1; no other model is promoted because of test results",
        "latency": "no latency SLO exists for this project; measured costs are reported, not used as a gate",
    }


def split(run, log) -> dict:
    cfg = run.config()
    ensure_prepared(run, log)
    prep = load_prepared(cfg)
    cat = prep.cat
    positives, sizes = load_positives(run)
    users_all = select_users(sizes, cfg)
    level_users = users_all[users_all.in_smoke] if cfg["level"] == "smoke" else users_all
    queries = build_queries(cat, level_users, positives, cfg)
    pairs = sample_pairs(cat, queries, positives, cfg)
    users_all.to_parquet(run.work / "users_split.parquet", index=False)
    queries.to_parquet(run.work / "query_manifest.parquet", index=False)
    pairs.to_parquet(run.work / "sampled_pairs.parquet", index=False)
    q_counts = queries.groupby(["partition", "group"]).size().unstack(fill_value=0).to_dict()
    u_counts = level_users.groupby(["partition", "group"]).size().unstack(fill_value=0).to_dict()
    overlap = {f"{a}&{b}": int(len(set(level_users[level_users.partition == a].offline_user_id)
                                   & set(level_users[level_users.partition == b].offline_user_id)))
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    manifest = {
        "created_at": now(), "level": cfg["level"], "seeds": {"split_seed": cfg["split"]["split_seed"],
                                                               "negative_seed": cfg["negatives"]["negative_seed"]},
        "catalog_hash": cat.catalog_hash,
        "main_split": {"users": int(len(users_all)), "users_sha256": sha256_df(users_all),
                       "by_group_partition": users_all.groupby(["group", "partition"]).size().to_dict(),
                       "by_bucket": users_all.groupby(["bucket", "partition"]).size().to_dict()},
        "level_users": {"users": int(len(level_users)), "by_partition_group": u_counts,
                        "nested_in_main_split": bool(cfg["level"] != "smoke" or level_users.in_smoke.all())},
        "partition_overlap_users": overlap,
        "queries": {"rows": int(len(queries)), "by_partition_group": q_counts,
                    "query_manifest_sha256": sha256_df(queries.drop(columns=["visible"]).assign(
                        visible_sha=[hashlib.sha256(v.tobytes()).hexdigest()[:16] for v in queries.visible])),
                    "rules": {"train": ["R0"], "val": ["R0", "R1", "R2", "R3"], "test": ["R0", "R1", "R2", "R3"]},
                    "r1_keyword_missing": int(queries.r1_keyword.isna().sum())},
        "pairs": {"rows": int(len(pairs)), "by_partition_label": pairs.groupby(["partition", "label"]).size().to_dict(),
                  "negatives_shortfall": int(pairs.attrs.get("shortfall", 0)), "sha256": sha256_df(pairs)},
        "files": {"users_split": "work/users_split.parquet", "query_manifest": "work/query_manifest.parquet",
                  "sampled_pairs": "work/sampled_pairs.parquet"},
    }
    write_json(run.path("split_manifest.json"), _jsonable(manifest))
    if not run.exists("selection_protocol.json"):
        write_json(run.path("selection_protocol.json"), selection_protocol(cfg))
    log(f"[split] users={len(level_users)} queries={len(queries)} pairs={len(pairs)} overlap={overlap}")
    return manifest


def _jsonable(obj):
    if isinstance(obj, dict):
        return {(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in obj.items()}
    return obj


def load_queries(run) -> pd.DataFrame:
    return pd.read_parquet(run.work / "query_manifest.parquet")


def load_pairs(run) -> pd.DataFrame:
    return pd.read_parquet(run.work / "sampled_pairs.parquet")
