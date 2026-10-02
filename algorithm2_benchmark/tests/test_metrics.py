"""Metric definitions of the algorithm 2 experiment, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from algorithm2_benchmark.pipeline.metrics import paired_bootstrap, query_metrics, user_macro


def test_metrics_match_hand_computed_values():
    ids = np.arange(1, 21)
    s = np.linspace(20, 1, 20)                                   # position p has rank p + 1
    m = query_metrics(s, ids, 0)
    assert (m["rank"], m["hit"], m["ndcg"], m["mrr"]) == (1, 1.0, 1.0, 1.0)
    m = query_metrics(s, ids, 7)
    assert m["rank"] == 8 and m["hit"] == 1.0 and m["ndcg"] == pytest.approx(1 / np.log2(9))
    m = query_metrics(s, ids, 8)
    assert m["rank"] == 9 and m["hit"] == 0.0 and m["ndcg"] == 0.0 and m["mrr"] == 0.0
    m = query_metrics(np.array([1.0, 1.0, 0.0]), np.array([1, 2, 3]), 0)
    assert m["auc"] == pytest.approx((1 + 0.5 * 1) / 2)          # a raw-score tie counts one half
    assert query_metrics(np.array([3.0]), np.array([1]), 0)["auc"] is None   # one candidate: undefined


def test_tie_break_is_deterministic_and_auc_ignores_ids():
    s = np.array([1.0, 1.0, 1.0, 0.5])
    a = query_metrics(s, np.array([20, 10, 30, 40]), 0)          # the target ties with ids 10 and 30
    b = query_metrics(s, np.array([5, 10, 30, 40]), 0)
    assert (a["rank"], b["rank"]) == (2, 1)                      # ties are broken by ascending id ...
    assert a["auc"] == b["auc"] == pytest.approx((1 + 0.5 * 2) / 3)   # ... but never count as discrimination


def test_user_macro_and_paired_bootstrap_resample_users():
    users = np.array([1, 1, 2])
    assert user_macro(users, np.array([1.0, 0.0, 1.0])) == (pytest.approx(0.75), 2)   # users weigh equally
    r = paired_bootstrap({1: 1.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 1.0, 3: 0.0}, reps=500, seed=42)
    assert r["users"] == 3 and r["mean_diff"] == pytest.approx(1 / 3) and r["ci95"][0] <= r["ci95"][1]
