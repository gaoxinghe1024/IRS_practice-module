import math

import pytest

from evaluation.metrics import (
    auc_pos_vs_neg,
    hits_at_k,
    mrr_at_k,
    ndcg_at_k,
    ndcg_binary,
    paired_bootstrap_diff,
    recall_at_k,
)


def test_hits_is_binary_and_recall_is_coverage():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert hits_at_k(ranked, rel, 2) == 1.0          # at least one hit
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert hits_at_k(ranked, {"z"}, 4) == 0.0
    assert math.isnan(recall_at_k(ranked, set(), 3))


def test_mrr_and_ndcg():
    assert mrr_at_k(["x", "a", "b"], {"b"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"] * 11 + ["b"], {"b"}, 10) == 0.0
    assert ndcg_binary(["a", "x"], {"a"}, 2) == 1.0
    assert ndcg_binary(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([3, 0, 1], [3, 1], 3) == pytest.approx((7 + 1 / 2) / (7 + 1 / math.log2(3)))


def test_auc_ties_and_paired_bootstrap():
    assert auc_pos_vs_neg([2, 2], [1, 2]) == pytest.approx(0.75)
    d = paired_bootstrap_diff([1, 1, 0, 1], [0, 1, 0, 0])
    assert d["mean_diff"] == 0.5 and d["n"] == 4
