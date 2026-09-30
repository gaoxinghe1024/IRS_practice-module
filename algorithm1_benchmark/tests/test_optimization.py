import unittest
from collections import Counter

import numpy as np
from scipy.special import expit

from kbbench.common import ROOT, read_json, read_rows, sha256
from kbbench.features import normalized_semantic
from kbbench.fit_candidates import fit_monotone
from kbbench.optimize import calibrate_cosines
from kbbench.representations import describe, normalize_vectors


class OptimizationUnitTests(unittest.TestCase):
    def test_templates_preserve_five_fields_and_ignore_forbidden_fields(self):
        item = dict(genres=["Action", "Sci-Fi"], themes=["Space"], demographic="Shounen", studio="Studio, Inc.", episodes=26)
        poison = dict(item, title="FORBIDDEN_TITLE", synopsis="FORBIDDEN_SYNOPSIS", rating=10, user_id=1, relations=[9])
        for style in ("original", "compact", "natural"):
            text = describe(item, style)
            self.assertEqual(text, describe(poison, style))
            for value in ("Action", "Sci-Fi", "Space", "Shounen", "Studio, Inc.", "26"):
                self.assertIn(value, text)

    def test_missing_fields_are_omitted_without_inventing_facts(self):
        item = dict(genres=["Drama"], themes=[], demographic=None, studio=None, episodes=None)
        self.assertEqual(describe(item, "compact"), "Drama.")
        self.assertNotIn("unknown", describe(item, "natural"))
        self.assertNotIn("episodes", describe(item, "natural"))

    def test_centering_removes_shared_offset_and_preserves_dimension(self):
        vectors = np.array([[3, 1, 0], [3, -1, 0], [3, 0, 1]], dtype=np.float32)
        normalized, mean = normalize_vectors(vectors, centered=True)
        self.assertEqual(normalized.shape, vectors.shape)
        np.testing.assert_allclose(mean, [3, 0, 1 / 3], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1, atol=1e-7)
        self.assertTrue(np.all(normalized[:, 0] == 0))

    def test_degenerate_centering_rejected(self):
        with self.assertRaises(ValueError):
            normalize_vectors(np.ones((2, 384)), centered=True)

    def test_semantic_calibration_is_bounded_and_keeps_order(self):
        scores = normalized_semantic(np.array([-1, 0.2, 0.5, 0.8, 1]), "corpus_quantile", dict(low=0.2, high=0.8))
        np.testing.assert_allclose(scores, [0, 0, 0.5, 1, 1])
        with self.assertRaises(ValueError):
            normalized_semantic([0.3], "corpus_quantile", dict(low=1, high=1))

    def test_corpus_calibration_reproducible_without_labels(self):
        rng = np.random.default_rng(7)
        vectors, _ = normalize_vectors(rng.normal(size=(20, 384)))
        self.assertEqual(calibrate_cosines(vectors), calibrate_cosines(vectors))

    def test_constrained_logistic_is_monotone(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 20, dtype=float)
        y = np.array([0, 0, 1, 1] * 20)
        coef, intercept, _ = fit_monotone(x, y)
        self.assertTrue(np.all(coef >= 0))
        self.assertGreater(expit(coef @ [1, 0] + intercept), expit(intercept))


class OptimizationArtifactTests(unittest.TestCase):
    def test_expanded_cohort_is_new_and_split_before_selection(self):
        protocol = read_json(ROOT / "data/optimization_protocol.json")
        cohort = list(read_rows(ROOT / "data/targets.jsonl"))
        original = {row["anime_id"] for row in read_rows(ROOT / "history/v1/data/targets.jsonl")}
        historical = {row["anime_id"] for row in cohort if row["split"] == "historical"}
        validation = {row["anime_id"] for row in cohort if row["split"] == "validation"}
        test = {row["anime_id"] for row in cohort if row["split"] == "test"}
        self.assertEqual(historical, original)
        self.assertEqual((len(historical), len(validation), len(test)), (120, 200, 400))
        self.assertFalse(validation & test or historical & (validation | test))
        self.assertEqual(protocol["targets_sha256"], sha256(ROOT / "data/targets.jsonl"))

    def test_validation_selection_never_uses_fresh_test_targets(self):
        selection = read_json(ROOT / "models/optimization_selection.json")
        cohort = list(read_rows(ROOT / "data/targets.jsonl"))
        validation = {row["anime_id"] for row in cohort if row["split"] == "validation"}
        self.assertTrue(set(selection["validation_target_ids"]) <= validation)
        self.assertFalse(selection["test_rankings_used_for_selection"])
        for row in read_rows(ROOT / "results/validation_trial_metrics.jsonl"):
            self.assertEqual(row["split"], "validation")
            self.assertIn(row["target_id"], validation)
        self.assertEqual(selection["validation_truth_sha256"], sha256(ROOT / "results/selection_ground_truth.jsonl"))
        self.assertEqual(selection["protocol_sha256"], sha256(ROOT / "data/optimization_protocol.json"))

    def test_both_training_sets_exclude_every_evaluation_target(self):
        cohort = {row["anime_id"] for row in read_rows(ROOT / "data/targets.jsonl")}
        endpoints = {}
        positive_sets = []
        for filename in ("training_pairs.jsonl.gz", "training_pairs_marginal.jsonl.gz"):
            rows = list(read_rows(ROOT / "data" / filename))
            self.assertEqual(Counter(row["label"] for row in rows), {0: 50000, 1: 50000})
            self.assertEqual(len({(row["left_id"], row["right_id"]) for row in rows}), 100000)
            self.assertFalse(cohort & {row[key] for row in rows for key in ("left_id", "right_id")})
            positive_sets.append({(row["left_id"], row["right_id"]) for row in rows if row["label"] == 1})
            endpoints[filename] = Counter(row[key] for row in rows if row["label"] == 0 for key in ("left_id", "right_id"))
        self.assertEqual(positive_sets[0], positive_sets[1])
        positive = Counter(aid for pair in positive_sets[0] for aid in pair)
        def distance(counts):
            keys = set(positive) | set(counts)
            return sum(abs(positive[key] - counts[key]) for key in keys) / 100000
        self.assertLess(distance(endpoints["training_pairs_marginal.jsonl.gz"]), distance(endpoints["training_pairs.jsonl.gz"]))

    def test_selected_models_are_from_registered_candidates(self):
        protocol = read_json(ROOT / "data/optimization_protocol.json")
        selection = read_json(ROOT / "models/optimization_selection.json")
        for family in ("B", "C", "D"):
            self.assertIn(selection[family], protocol[family + "_candidates"])
        for filename, expected in selection["files"].items():
            self.assertEqual(sha256(ROOT / filename), expected)
        metadata = read_json(ROOT / "models/bge_optimized_metadata.json")
        self.assertEqual(metadata["dimensions"], 384)
        self.assertEqual(metadata["model"], "BAAI/bge-small-en-v1.5")

    def test_before_after_share_targets_truth_and_fixed_baseline(self):
        before = list(read_rows(ROOT / "results/baseline/per_query_metrics.jsonl"))
        after = list(read_rows(ROOT / "results/per_query_metrics.jsonl"))
        key = lambda row: (row["source"], row["split"], row["target_id"], row["method"])
        before = {key(row): row for row in before}
        after = {key(row): row for row in after}
        self.assertEqual(set(before), set(after))
        for identity in before:
            self.assertEqual(before[identity]["relevant_ids"], after[identity]["relevant_ids"])
            if identity[-1] == "A":
                self.assertEqual(before[identity]["predicted_ids"], after[identity]["predicted_ids"])


if __name__ == "__main__":
    unittest.main()
