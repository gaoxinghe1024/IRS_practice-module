"""Offline evaluation context: models and data loaded straight from versioned artifacts.

All statistics come from the ranking-protocol train split (train interaction
counts, user means, neighbour lists, histories). Evaluation users are sampled
with a fixed seed from users that have a held-out positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.recommendation.constraints import ConstraintEngine, ConstraintSet
from app.recommendation.features import CatalogSnapshot, build_snapshot
from app.recommendation.similarity.weighted import WeightedAttributeScorer
from pipelines.common import latest_dataset_version, read_json
from pipelines.fit import artifact_dir
from pipelines.split import load_split


@dataclass
class EvalContext:
    settings: Settings
    version: str
    art: Path
    manifest: dict
    snap: CatalogSnapshot
    pool_mask: np.ndarray                    # served & recommendable (the candidate universe)
    cf_items: np.ndarray
    cat_to_cf: np.ndarray
    offline_users: np.ndarray
    user_index: dict
    icf: object = None
    mf: object = None
    ncf: object = None
    weighted: WeightedAttributeScorer = None
    engine: ConstraintEngine = None
    _train: pd.DataFrame = field(default=None, repr=False)

    def train(self) -> pd.DataFrame:
        if self._train is None:
            self._train = load_split(self.settings, "ranking", "train", self.version)
        return self._train

    def histories(self, users: list[int], split: pd.DataFrame | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df = (split if split is not None else self.train())
        sub = df[df.offline_user_id.isin(set(users))]
        out = {}
        for u, g in sub.groupby("offline_user_id", sort=False):
            out[int(u)] = (g.item_idx.to_numpy(np.int64), g.rating.to_numpy(np.float64))
        return out

    def history_tags(self, items: np.ndarray) -> set[tuple[str, str]]:
        out = set()
        for ns in ("genre", "theme"):
            tm = self.snap.tags.get(ns)
            if tm is not None:
                for i in items:
                    out.update((ns, t) for t in tm.tags_of(int(i)))
        return out


def load_context(settings: Settings, load_models: bool = True) -> EvalContext:
    version = latest_dataset_version(settings)
    art = artifact_dir(settings, version)
    manifest = read_json(art / "manifest.json")
    snap = build_snapshot(pd.read_parquet(art / "catalog" / "catalog.parquet"),
                          pd.read_parquet(art / "catalog" / "item_tags.parquet"), version, settings.mode)
    snap.with_train_interactions(np.load(art / "stats" / "train_interactions.npy"))
    engine = ConstraintEngine(snap)
    pool = engine.evaluate(ConstraintSet()).mask
    cf_items = np.load(art / "models" / "cf_items.npy")
    cat_to_cf = np.full(snap.n, -1, dtype=np.int64)
    cat_to_cf[cf_items] = np.arange(len(cf_items))
    users = np.load(art / "models" / "offline_users.npy")
    ctx = EvalContext(settings, version, art, manifest, snap, pool, cf_items, cat_to_cf, users,
                      {int(u): n for n, u in enumerate(users)},
                      weighted=WeightedAttributeScorer(settings.recommendation.get("similar", {}).get("weights"),
                                                       settings.recommendation.get("similar", {}).get("tau_episodes", 12.0)),
                      engine=engine)
    if load_models:
        comps = manifest.get("components", {})
        if comps.get("icf", {}).get("status") == "trained":
            from app.recommendation.recall.icf import ICFModel, ICFScorer

            icf_cfg = settings.recommendation.get("icf", {})
            ctx.icf = ICFScorer(ICFModel.load(art / "models" / "icf.npz"),
                                damping=float(icf_cfg.get("prediction_damping", 0.0)),
                                prior_weight=float(icf_cfg.get("user_mean_prior_weight", 0.0)))
        if comps.get("mf", {}).get("status") == "trained":
            from app.recommendation.recall.matrix_factorization import MFModel

            ctx.mf = MFModel.load(art / "models" / "mf.npz")
        if comps.get("ncf", {}).get("status") == "trained":
            from app.recommendation.recall.ncf import NCFModel

            ctx.ncf = NCFModel.load(art / "models" / "ncf.pt")
    return ctx


def sample_users(df: pd.DataFrame, n: int | None, seed: int) -> list[int]:
    users = np.sort(df.offline_user_id.unique())
    if n is None or n >= len(users):
        return users.tolist()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(users, size=n, replace=False)).tolist()


def engine_scores(ctx: EvalContext, engine: str, user: int, hist_items: np.ndarray, hist_r: np.ndarray,
                  candidates_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ranked (catalogue ids, scores) for one offline user; only eligible candidates, pure engine output."""
    snap = ctx.snap
    if engine == "popularity":
        ids = np.flatnonzero(candidates_mask)
        s = snap.train_interactions[ids].astype(np.float64)
    elif engine == "icf":
        cf_hist = ctx.cat_to_cf[hist_items]
        keep = cf_hist >= 0
        pred = ctx.icf.predict(cf_hist[keep], hist_r[keep], user_mean=float(hist_r.mean()), n_ratings=len(hist_r))
        ids = ctx.cf_items[pred.items]
        ok = candidates_mask[ids]
        ids, s = ids[ok], pred.predicted[ok] + 1e-6 * np.log1p(pred.support[ok])   # support breaks exact ties
    elif engine in ("mf", "ncf"):
        model = ctx.mf if engine == "mf" else ctx.ncf
        u = ctx.user_index.get(int(user))
        if model is None or u is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        allp = model.predict_user(u)
        ids = ctx.cf_items
        ok = candidates_mask[ids]
        ids, s = ids[ok], np.asarray(allp)[ok]
    else:
        raise ValueError(engine)
    order = np.lexsort((snap.keys[ids], -s))
    return ids[order], s[order]
