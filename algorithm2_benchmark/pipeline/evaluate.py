"""Full-catalogue evaluator for algorithm 2 (strict mode: no relaxation, ever).

For each query: build the profile from H, take the R0 candidates (catalogue members not in H, in
anime_id order), score them once per scorer, then evaluate every rule scenario on its subset (a score
depends only on the query and the item, so R1-R3 reuse the R0 scores). Rule masks come from the same
ConstraintEngine the service uses. A target that fails a rule is not inserted: the query is recorded
as `target_not_eligible` and excluded from that scenario's quality metrics.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from algorithm2_benchmark.pipeline.metrics import K, query_metrics
from app.recommendation.algorithm2.catalog import Alg2Catalog
from app.recommendation.algorithm2.features import compute_features
from app.recommendation.algorithm2.profiles import build_profile
from app.recommendation.algorithm2.scorers import QueryContext, top_k
from app.recommendation.constraints import ConstraintEngine, ConstraintSet

RULE_IDS = ("R0", "R1", "R2", "R3")


class RuleMasks:
    """Strict rule masks over the snapshot, computed by the shared ConstraintEngine (cached per keyword)."""

    def __init__(self, cat: Alg2Catalog, engine: ConstraintEngine, cfg: dict):
        self.cat, self.engine = cat, engine
        r2 = cfg["rules"]["R2"]
        self.r2_spec = {"min_score": float(r2["min_score"]), "max_episodes": int(r2["max_episodes"])}
        self.base = engine.evaluate(ConstraintSet(require_recommendable=False, require_scored=True)).mask
        if not np.array_equal(self.base, cat.member):
            raise RuntimeError("rule-layer base pool differs from the algorithm-2 catalogue membership")
        self.r2 = engine.evaluate(ConstraintSet(require_recommendable=False, require_scored=True,
                                                **self.r2_spec)).mask
        self._kw: dict[str, np.ndarray] = {}

    def constraint_set(self, rule: str, keyword: str | None) -> ConstraintSet | None:
        cs = ConstraintSet(require_recommendable=False, require_scored=True)
        if rule in ("R1", "R3"):
            if not keyword:
                return None
            cs.keywords = [keyword]
        if rule in ("R2", "R3"):
            cs.min_score, cs.max_episodes = self.r2_spec["min_score"], self.r2_spec["max_episodes"]
        return cs

    def keyword(self, kw: str) -> np.ndarray:
        m = self._kw.get(kw)
        if m is None:
            m = self.engine.evaluate(ConstraintSet(require_recommendable=False, require_scored=True,
                                                   keywords=[kw])).mask
            self._kw[kw] = m
        return m

    def mask(self, rule: str, keyword: str | None) -> np.ndarray | None:
        if rule == "R0":
            return self.base
        if rule == "R2":
            return self.r2
        if not keyword:
            return None
        return self.keyword(keyword) if rule == "R1" else (self.keyword(keyword) & self.r2)

    def spec(self, rule: str, keyword: str | None) -> dict:
        out = {"rule": rule, "keywords": [keyword] if rule in ("R1", "R3") and keyword else []}
        if rule in ("R2", "R3"):
            out.update(self.r2_spec)
        return out


def r0_candidates(cat: Alg2Catalog, history: np.ndarray) -> np.ndarray:
    m = cat.members
    return m[~np.isin(m, history)] if len(history) else m


def ids_hash(anime_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(anime_ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def evaluate_queries(cat: Alg2Catalog, rules: RuleMasks, queries: pd.DataFrame, scorers: dict, rule_ids=RULE_IDS,
                     k: int = K, keep_top: bool = True, log=None, timing: dict | None = None) -> pd.DataFrame:
    """One row per (query, rule, scorer). `scorers` maps a model label to a scorer object."""
    cols: dict[str, list] = {c: [] for c in (
        "query_id", "user", "collection_id", "group", "partition", "history_size", "rule", "keyword", "model",
        "target_key", "n_cand", "cand_hash", "target_eligible", "excluded_reason", "target_score", "rank", "hit",
        "ndcg", "mrr", "auc", "top_keys", "top_scores", "violations", "dup_in_top", "saved_in_top", "relaxed")}
    t_start = time.time()
    timing = timing if timing is not None else {}
    for qi, q in enumerate(queries.itertuples(index=False)):
        H = np.asarray(q.visible, dtype=np.int64)
        prof = build_profile(cat, H)
        cand = r0_candidates(cat, H)
        ctx = QueryContext(cat, prof, cand)
        scores = {}
        for name, sc in scorers.items():
            t0 = time.perf_counter()
            scores[name] = sc.score(ctx)
            timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0
        t = int(q.target)
        hset = set(H.tolist())
        for rule in rule_ids:
            rmask = rules.mask(rule, q.r1_keyword)
            if rmask is None:
                sel = np.zeros(len(cand), dtype=bool)
                eligible, reason = False, "rule_not_applicable_no_genre_in_history"
            else:
                sel = rmask[cand]
                eligible = bool(rmask[t]) and t not in hset
                reason = None if eligible else "target_not_eligible"
            sub = cand[sel]
            n = int(len(sub))
            if reason is None and n == 0:
                reason = "no_candidates"
            chash = ids_hash(cat.anime_id[sub])
            tpos = int(np.searchsorted(cat.anime_id[sub], cat.anime_id[t])) if (eligible and n) else -1
            for name, s_all in scores.items():
                s = s_all[sel]
                cols["query_id"].append(q.query_id)
                cols["user"].append(int(q.offline_user_id))
                cols["collection_id"].append(q.collection_id)
                cols["group"].append(q.group)
                cols["partition"].append(q.partition)
                cols["history_size"].append(int(q.history_size))
                cols["rule"].append(rule)
                cols["keyword"].append(q.r1_keyword if rule in ("R1", "R3") else None)
                cols["model"].append(name)
                cols["target_key"].append(q.target_key)
                cols["n_cand"].append(n)
                cols["cand_hash"].append(chash)
                cols["target_eligible"].append(eligible)
                cols["excluded_reason"].append(reason)
                cols["relaxed"].append(False)
                if reason is None:
                    m = query_metrics(s, cat.anime_id[sub], tpos, k)
                    cols["target_score"].append(float(s[tpos]))
                    cols["rank"].append(m["rank"])
                    cols["hit"].append(m["hit"])
                    cols["ndcg"].append(m["ndcg"])
                    cols["mrr"].append(m["mrr"])
                    cols["auc"].append(m["auc"])
                else:
                    for c in ("target_score", "rank", "hit", "ndcg", "mrr", "auc"):
                        cols[c].append(None)
                top = top_k(s, sub, cat.anime_id, k) if n else np.zeros(0, dtype=np.int64)
                top_items = sub[top]
                viol = int((~rmask[top_items]).sum()) if rmask is not None else 0
                viol += int(sum(1 for i in top_items.tolist() if i in hset))
                cols["violations"].append(viol)
                cols["dup_in_top"].append(int(len(top_items) - len(set(top_items.tolist()))))
                cols["saved_in_top"].append(int(sum(1 for i in top_items.tolist() if i in hset)))
                cols["top_keys"].append("|".join(cat.snap.keys[top_items].tolist()) if keep_top else None)
                cols["top_scores"].append("|".join(f"{x:.6g}" for x in s[top].tolist()) if keep_top else None)
        if log and (qi + 1) % 500 == 0:
            log(f"[evaluate] {qi + 1}/{len(queries)} queries ({time.time() - t_start:.0f}s)")
    return pd.DataFrame(cols)


# ------------------------------------------------------------------------------ validation cache


class FeatureCache:
    """R0 candidates and their eight features for a fixed list of queries (model-independent)."""

    def __init__(self, cat: Alg2Catalog, queries: pd.DataFrame, log=None):
        self.queries = queries.reset_index(drop=True)
        offsets = [0]
        cands, feats, tpos = [], [], []
        t0 = time.time()
        for qi, q in enumerate(self.queries.itertuples(index=False)):
            H = np.asarray(q.visible, dtype=np.int64)
            prof = build_profile(cat, H)
            cand = r0_candidates(cat, H)
            feats.append(compute_features(cat, prof, cand))
            cands.append(cand.astype(np.int32))
            tpos.append(int(np.searchsorted(cat.anime_id[cand], cat.anime_id[int(q.target)])))
            offsets.append(offsets[-1] + len(cand))
            if log and (qi + 1) % 1000 == 0:
                log(f"[cache] features for {qi + 1}/{len(self.queries)} queries ({time.time() - t0:.0f}s)")
        self.F = np.concatenate(feats) if feats else np.zeros((0, 8), dtype=np.float32)
        self.cand = np.concatenate(cands) if cands else np.zeros(0, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.tpos = np.asarray(tpos, dtype=np.int64)
        self.users = self.queries.offline_user_id.to_numpy()
        self.anime = cat.anime_id[self.cand]

    def metrics_from_scores(self, scores: np.ndarray, k: int = K) -> pd.DataFrame:
        rows = []
        for qi in range(len(self.queries)):
            a, b = self.offsets[qi], self.offsets[qi + 1]
            m = query_metrics(scores[a:b], self.anime[a:b], int(self.tpos[qi]), k)
            rows.append((self.users[qi], m["hit"], m["ndcg"], m["mrr"], m["auc"], m["n_cand"]))
        return pd.DataFrame(rows, columns=["user", "hit", "ndcg", "mrr", "auc", "n_cand"])


def primary_mask(df: pd.DataFrame) -> pd.Series:
    """The pre-registered primary set: rule R0, main group, more than 8 candidates, target eligible."""
    return (df["rule"] == "R0") & (df["group"] == "main") & (df["n_cand"] > K) & df["target_eligible"]
