import numpy as np
import pandas as pd

from app.recommendation.recall.matrix_factorization import MFModel, fit_mf, rmse


def _lowrank(seed=0, n_users=200, n_items=80, k=3, density=0.25):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
    R = 6.5 + U @ V.T + rng.normal(0, 0.3, (n_users, n_items))
    obs = rng.random((n_users, n_items)) < density
    u, i = np.nonzero(obs)
    return u, i, np.clip(np.round(R[u, i]), 1, 10), n_users, n_items


def test_mf_learns_low_rank_structure_and_folds_in(tmp_path):
    u, i, r, nu, ni = _lowrank()
    perm = np.random.default_rng(1).permutation(len(r))
    tr, va = perm[: int(0.85 * len(r))], perm[int(0.85 * len(r)):]
    model = fit_mf(u[tr], i[tr], r[tr], nu, ni, val=(u[va], i[va], r[va]), latent_dim=8, epochs=60,
                   batch_size=512, learning_rate=0.02, reg=0.02, patience=5, seed=0)
    baseline = float(np.sqrt(np.mean((r[va] - r[tr].mean()) ** 2)))
    assert rmse(model, u[va], i[va], r[va]) < 0.85 * baseline
    p, b = model.fold_in(i[u == 0][:10], r[u == 0][:10], reg=1.0, min_ratings=3)
    assert p.shape == (8,) and np.isfinite(b)
    assert model.fold_in(i[:2], r[:2], min_ratings=3) is None          # too few ratings -> caller falls back
    model.save(tmp_path / "mf.npz")
    back = MFModel.load(tmp_path / "mf.npz")
    assert np.allclose(back.predict_user(0), model.predict_user(0))


def test_ranking_split_holds_out_distinct_high_ratings(settings):
    from pipelines.split import load_split, split_manifest

    tr, va, te = (load_split(settings, "ranking", p) for p in ("train", "val", "test"))
    thr = settings.pipeline.get("positive_threshold", 8)
    assert (va.rating >= thr).all() and (te.rating >= thr).all()
    assert va.offline_user_id.is_unique and te.offline_user_id.is_unique
    key = lambda df: set(zip(df.offline_user_id, df.item_idx, strict=True))  # noqa: E731
    assert not (key(va) & key(te)) and not (key(tr) & key(va)) and not (key(tr) & key(te))
    m = split_manifest(settings, "ranking")
    assert m["counts"] == {"train": len(tr), "val": len(va), "test": len(te)} and len(m["fingerprint"]) == 16
    ex = pd.concat([load_split(settings, "explicit", p) for p in ("train", "val", "test")])
    assert len(ex) == len(tr) + len(va) + len(te)                      # same ratings, different protocol
