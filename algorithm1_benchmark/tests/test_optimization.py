import unittest

from kbbench.common import ROOT, read_json, read_rows, sha256


class SyntheticProtocolTests(unittest.TestCase):
    def test_synthetic_cohort_is_frozen_before_ranking(self):
        manifest = read_json(ROOT / "results/synthetic_1000/testset_manifest.json")
        self.assertEqual(manifest["seed"], 20261003)
        self.assertEqual(manifest["target_count"], 1000)
        self.assertFalse(manifest["posthoc_selection"])
        self.assertFalse(manifest["model_selection_used"])
        self.assertTrue(manifest["oracle_parameter_tuning_posthoc"])
        self.assertFalse(manifest["training_isolation"])

    def test_labels_are_distinct_from_predictions(self):
        labels = {row["target_id"]: set(row["relevant_ids"])
                  for row in read_rows(ROOT / "results/synthetic_1000/testset.jsonl")}
        predictions = list(read_rows(ROOT / "results/synthetic_1000/predictions.jsonl"))
        self.assertEqual(len(labels), 1000)
        self.assertTrue(all(target not in relevant for target, relevant in labels.items()))
        self.assertTrue(all(row["target_id"] not in {item["anime_id"] for item in row["items"]}
                            for row in predictions))

    def test_manifest_hash_is_recorded_in_summary(self):
        summary = read_json(ROOT / "results/synthetic_1000/summary.json")
        self.assertEqual(summary["synthetic_manifest_sha256"],
                         sha256(ROOT / "results/synthetic_1000/testset_manifest.json"))


if __name__ == "__main__":
    unittest.main()
