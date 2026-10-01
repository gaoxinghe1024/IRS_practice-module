"""Checks real delivered data and predictions; catches leakage and evidence mismatch."""
import gzip
import json
import unittest
from collections import defaultdict

import numpy as np

from kbbench.common import ROOT, read_json, read_rows, sha256
from kbbench.engine import SimilarityEngine
from kbbench.mal_html import parse_recommendations
from kbbench.metrics import ranking_metrics
from baseline import recommend as baseline_recommend


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
        cls.ids = {row["anime_id"] for row in cls.items}
        cls.cohort = {row["anime_id"] for row in read_rows(ROOT / "data/targets.jsonl")}

    def test_catalog_size_and_training_separation(self):
        self.assertEqual(len(self.items), len(self.ids))
        self.assertEqual(len(self.ids), 14202)
        pairs = list(read_rows(ROOT / "data/training_pairs.jsonl.gz"))
        endpoints = {row[field] for row in pairs for field in ("left_id", "right_id")}
        self.assertFalse(endpoints & self.cohort)
        self.assertTrue(endpoints <= self.ids)
        unique = {(row["left_id"], row["right_id"]) for row in pairs}
        self.assertEqual(len(unique), len(pairs))
        self.assertTrue(all(row["left_id"] < row["right_id"] for row in pairs))

    def test_ground_truth_matches_saved_votes(self):
        for label in read_rows(ROOT / "data/mal_ground_truth.jsonl"):
            path = ROOT / label["cache_file"]
            meta = read_json(path.with_name(path.name.replace(".json.gz", ".meta.json")))
            self.assertEqual(sha256(path), meta["sha256"])
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if meta.get("raw_page"):
                raw_path = path.with_name(meta["raw_page"])
                raw_meta = read_json(raw_path.with_name(raw_path.name.replace(".html.gz", ".html.meta.json")))
                self.assertEqual(sha256(raw_path), raw_meta["sha256"])
                with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
                    self.assertEqual(parse_recommendations(stream.read(), label["target_id"]), payload)
            votes = {}
            for row in payload["data"]:
                aid = int(row["entry"]["mal_id"])
                count = int(row.get("votes", 0))
                if aid in self.ids and aid != label["target_id"] and count > 0:
                    votes[aid] = max(count, votes.get(aid, 0))
            expected = sorted(votes, key=lambda aid: (-votes[aid], aid))[:5]
            self.assertEqual(expected, label["relevant_ids"])
            self.assertEqual([votes[aid] for aid in expected], label["votes"])

    def test_stored_metrics_are_reproducible(self):
        groups = defaultdict(list)
        for row in read_rows(ROOT / "results/per_query_metrics.jsonl"):
            for metric, value in ranking_metrics(row["predicted_ids"], row["relevant_ids"]).items():
                self.assertAlmostEqual(row[metric], value)
            groups[(row["source"], row["split"], row["method"])].append(row)
        for row in read_json(ROOT / "results/summary.json")["summaries"]:
            entries = groups[(row["source"], row["split"], row["method"])]
            self.assertEqual(len(entries), row["n_targets"])
            for metric in ("coverage_at_5", "hit_rate_at_5", "ndcg_at_5", "precision_at_5"):
                self.assertAlmostEqual(np.mean([entry[metric] for entry in entries]), row[metric])

    def test_recommendations_are_offline_and_match_artifact(self):
        engine = SimilarityEngine()
        stored = [row for row in read_rows(ROOT / "results/predictions.jsonl") if row["target_id"] == 1]
        self.assertEqual(len(stored), 4)
        for row in stored:
            computed = engine.recommend(1, row["method"])
            self.assertEqual([item["anime_id"] for item in computed], [item["anime_id"] for item in row["items"]])
            self.assertTrue(all(item["anime_id"] != 1 for item in computed))

    def test_zero_dependency_baseline_agrees_with_experiment(self):
        engine = SimilarityEngine()
        for target in [1, 1535, 5114, 9253]:
            self.assertEqual([row["anime_id"] for row in baseline_recommend(target)],
                             [row["anime_id"] for row in engine.recommend(target, "A")])



if __name__ == "__main__":
    unittest.main()
