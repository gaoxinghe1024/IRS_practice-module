import numpy as np

from app.recommendation.recall.icf import ICFScorer, build_matrices, exact_adjusted_cosine_dense, fit_icf


def _random_ratings(seed=0, n_users=40, n_items=12, density=0.55):
    rng = np.random.default_rng(seed)
    obs = rng.random((n_users, n_items)) < density
    R = rng.integers(1, 11, size=(n_users, n_items)).astype(float)
    # a user whose ratings are all identical -> every centred value is exactly 0
    obs[0, :6] = True
    obs[0, 6:] = False
    R[0, :6] = 7
    return R, obs


def _fit(R, obs, **kw):
    u, i = np.nonzero(obs)
    return fit_icf(u, i, R[u, i], R.shape[0], R.shape[1], top_n=10_000, block_size=5, **kw)


def test_icf_matches_bruteforce_oracle_no_shrinkage():
    R, obs = _random_ratings()
    model = _fit(R, obs, min_co_rated=3, shrinkage=0.0)
    oracle = exact_adjusted_cosine_dense(R, obs, min_co_rated=3, shrinkage=0.0)
    got = model.S.toarray()
    expected = np.where(oracle > 0, oracle, 0.0)       # recall keeps positive similarities
    assert np.allclose(got, expected, atol=1e-5)


def test_icf_matches_oracle_with_shrinkage():
    R, obs = _random_ratings(seed=3)
    model = _fit(R, obs, min_co_rated=3, shrinkage=10.0)
    oracle = exact_adjusted_cosine_dense(R, obs, min_co_rated=3, shrinkage=10.0)
    assert np.allclose(model.S.toarray(), np.where(oracle > 0, oracle, 0.0), atol=1e-5)


def test_centred_zero_ratings_still_count_as_co_rated():
    R, obs = _random_ratings()
    u, i = np.nonzero(obs)
    A, M, mu = build_matrices(u, i, R[u, i], R.shape[0], R.shape[1])
    # user 0 has 6 observed ratings that all centre to exactly 0: they stay in A (explicit zeros) and M
    assert M[0].nnz == 6
    assert A[0].nnz == 6 and np.allclose(A[0].data, 0)
    co = (M.T @ M).toarray()
    model = _fit(R, obs, min_co_rated=3, shrinkage=0.0)
    nz = model.S.nonzero()
    assert np.array_equal(model.co_counts.toarray()[nz], co[nz])


def test_co_count_threshold_drops_pairs():
    R, obs = _random_ratings(seed=5)
    model = _fit(R, obs, min_co_rated=12, shrinkage=0.0)
    co = (obs.T.astype(int) @ obs.astype(int))
    S = model.S.toarray()
    assert np.all(S[co < 12] == 0)


def test_prediction_formula_matches_manual_computation():
    R, obs = _random_ratings(seed=7)
    model = _fit(R, obs, min_co_rated=3, shrinkage=0.0)
    scorer = ICFScorer(model)
    rated = np.array([0, 2, 5])
    vals = np.array([9.0, 4.0, 7.0])
    mu = vals.mean()
    pred = scorer.predict(rated, vals)
    S = model.S.toarray()
    for item, p in zip(pred.items, pred.predicted, strict=True):
        num = sum(S[item, j] * (r - mu) for j, r in zip(rated, vals, strict=True))
        den = sum(abs(S[item, j]) for j in rated)
        assert np.isclose(p, np.clip(mu + num / den, 1, 10))
    # items without any rated neighbour get no prediction
    no_support = [i for i in range(R.shape[1]) if all(S[i, j] == 0 for j in rated)]
    assert not set(no_support) & set(pred.items.tolist())


def test_low_rating_contributes_negatively():
    R, obs = _random_ratings(seed=11)
    model = _fit(R, obs, min_co_rated=3, shrinkage=0.0)
    scorer = ICFScorer(model)
    high = scorer.predict(np.array([1, 3]), np.array([10.0, 5.0]))
    low = scorer.predict(np.array([1, 3]), np.array([5.0, 10.0]))
    S = model.S.toarray()
    for it in set(high.items) & set(low.items):
        a = high.predicted[list(high.items).index(it)]
        b = low.predicted[list(low.items).index(it)]
        if S[it, 1] > S[it, 3] + 1e-6:
            assert a >= b - 1e-9


def test_save_and_load_roundtrip(tmp_path):
    R, obs = _random_ratings()
    model = _fit(R, obs)
    model.save(tmp_path / "icf.npz")
    from app.recommendation.recall.icf import ICFModel

    back = ICFModel.load(tmp_path / "icf.npz")
    assert np.allclose(back.S.toarray(), model.S.toarray())
    assert np.allclose(back.co_counts.toarray(), model.co_counts.toarray())
