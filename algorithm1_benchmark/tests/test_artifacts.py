"""Checks for the published synthetic 1,000-query artifact set."""
import unittest
from collections import defaultdict

import numpy as np

from kbbench.common import ROOT, read_json, read_rows, sha256
from kbbench.engine import SimilarityEngine
from kbbench.metrics import ranking_metrics


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "results/synthetic_1000"
        cls.summary = read_json(cls.output / "summary.json")
        cls.manifest = read_json(cls.output / "testset_manifest.json")

    def test_manifest_freezes_synthetic_protocol(self):
        self.assertEqual(self.manifest["target_count"], 1000)
        self.assertEqual(self.manifest["label_source"], "synthetic_metadata_oracle")
        self.assertFalse(self.manifest["posthoc_selection"])
        self.assertFalse(self.manifest["model_selection_used"])
        self.assertEqual(self.summary["synthetic_manifest_sha256"],
                         sha256(self.output / "testset_manifest.json"))

    def test_each_target_has_five_labels_and_each_method_ranking(self):
        labels = list(read_rows(self.output / "testset.jsonl"))
        predictions = list(read_rows(self.output / "predictions.jsonl"))
        self.assertEqual(len(labels), 1000)
        self.assertEqual(len({row["target_id"] for row in labels}), 1000)
        self.assertTrue(all(len(row["relevant_ids"]) == 5 for row in labels))
        self.assertEqual(len(predictions), 4000)
        self.assertEqual({row["method"] for row in predictions}, {"A", "B", "C", "D"})
        self.assertTrue(all(len(row["items"]) == 5 for row in predictions))

    def test_stored_metrics_are_reproducible(self):
        groups = defaultdict(list)
        for row in read_rows(self.output / "per_query_metrics.jsonl"):
            for metric, value in ranking_metrics(row["predicted_ids"], row["relevant_ids"]).items():
                self.assertAlmostEqual(row[metric], value)
            groups[row["method"]].append(row)
        for row in self.summary["summaries"]:
            entries = groups[row["method"]]
            self.assertEqual(len(entries), row["n_targets"])
            for metric in ("coverage_at_5", "hit_rate_at_5", "ndcg_at_5", "precision_at_5"):
                self.assertAlmostEqual(np.mean([entry[metric] for entry in entries]), row[metric])

    def test_recommendations_replay_offline(self):
        engine = SimilarityEngine()
        target_id = next(iter(read_rows(self.output / "testset.jsonl")))["target_id"]
        stored = [row for row in read_rows(self.output / "predictions.jsonl")
                  if row["target_id"] == target_id]
        self.assertEqual(len(stored), 4)
        for row in stored:
            computed = engine.recommend(target_id, row["method"])
            self.assertEqual([item["anime_id"] for item in computed],
                             [item["anime_id"] for item in row["items"]])
            self.assertTrue(all(item["anime_id"] != target_id for item in computed))


if __name__ == "__main__":
    unittest.main()
