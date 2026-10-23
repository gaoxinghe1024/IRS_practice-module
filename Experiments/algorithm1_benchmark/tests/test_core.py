import math
import unittest

import numpy as np

from kbbench.features import WEIGHTS, cosine_to_unit, jaccard, pair_features, rank_top_k, semantic_text
from kbbench.metrics import ranking_metrics, replacement_rule


def item(**overrides):
    result = dict(anime_id=1, title="Example", genres=["Action", "Sci-Fi"], themes=["Space"],
                  demographic="Shounen", studio="Studio, Incorporated", episodes=12)
    result.update(overrides)
    return result


class FeatureTests(unittest.TestCase):
    def test_perfect_match(self):
        np.testing.assert_equal(pair_features(item(), item()), [1, 1, 1, 1, 1])
        self.assertAlmostEqual(np.dot(pair_features(item(), item()), WEIGHTS), 1)

    def test_set_duplicates_do_not_change_overlap(self):
        self.assertEqual(jaccard(["Action", "Action"], ["Action", "Drama"]), 0.5)
        self.assertEqual(jaccard([], []), 0)

    def test_single_label_is_not_split_at_comma(self):
        self.assertEqual(pair_features(item(), item(studio="Studio"))[3], 0)
        self.assertEqual(pair_features(item(), item(studio="STUDIO, INCORPORATED"))[3], 1)

    def test_unknowns_do_not_create_tag_matches(self):
        missing = item(genres=[], themes=[], demographic=None, studio=None, episodes=None)
        self.assertEqual(pair_features(missing, missing), (0, 0, 0, 0, 0.5))

    def test_numeric_episode_distance(self):
        self.assertAlmostEqual(pair_features(item(), item(episodes=24))[4], 0.5)

    def test_only_five_dimensions_in_embedding_text(self):
        a, b = item(), item(anime_id=999, title="Different", score=10, synopsis="extra", relations=[1])
        self.assertEqual(semantic_text(a), semantic_text(b))

    def test_unseen_item_needs_no_user_history(self):
        new_item = item(anime_id=999999, title="New title")
        self.assertEqual(pair_features(new_item, item()), (1, 1, 1, 1, 1))

    def test_cosine_normalization_is_global_not_query_relative(self):
        np.testing.assert_equal(cosine_to_unit(np.array([-1, 0, 1])), [0, 0.5, 1])


class RankingTests(unittest.TestCase):
    def test_ties_and_self_exclusion(self):
        ids = np.array([9, 1, 4, 3])
        order = rank_top_k(ids, [1, 1, 1, 1], 1, 5)
        self.assertEqual(ids[order].tolist(), [3, 4, 9])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            rank_top_k([1, 2], [1, float("nan")], 1)

    def test_recall_and_hit_rate_are_different(self):
        result = ranking_metrics([10, 20, 30, 40, 50], [10, 99, 98])
        self.assertAlmostEqual(result["coverage_at_5"], 1 / 3)
        self.assertEqual(result["hit_rate_at_5"], 1)
        self.assertAlmostEqual(result["ndcg_at_5"], 1 / (1 + 1 / math.log2(3) + 0.5))

    def test_perfect_ranking_and_empty_truth(self):
        self.assertEqual(ranking_metrics([1, 2, 3], [1, 2, 3])["ndcg_at_5"], 1)
        with self.assertRaises(ValueError):
            ranking_metrics([1], [])

    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1, 1, 2], [1, 2])

    def test_two_percentage_point_rule_is_conjunctive(self):
        base = dict(coverage_at_5=0.30, ndcg_at_5=0.40)
        self.assertTrue(replacement_rule(base, dict(coverage_at_5=0.32, ndcg_at_5=0.40))["passes"])
        self.assertFalse(replacement_rule(base, dict(coverage_at_5=0.319, ndcg_at_5=0.50))["passes"])
        self.assertFalse(replacement_rule(base, dict(coverage_at_5=0.35, ndcg_at_5=0.39))["passes"])

if __name__ == "__main__":
    unittest.main()
