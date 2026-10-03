"""Common recommendation interface for the four replaceable similarity modules."""
import numpy as np
from scipy.special import expit

from .common import ROOT, read_json, read_rows, sha256
from .features import DIMENSIONS, WEIGHTS, normalized_semantic, feature_matrix, rank_top_k

METHODS = {"A": "Weighted five-dimension similarity", "B": "Logistic regression",
           "C": "BGE embeddings", "D": "0.6 tag + 0.4 semantic"}


class SimilarityEngine:
    def __init__(self, root=ROOT, variant="optimized"):
        if variant != "optimized":
            raise ValueError("Only the optimized release is included in this benchmark")
        self.items = list(read_rows(root / "data/catalog.jsonl.gz"))
        self.ids = np.asarray([item["anime_id"] for item in self.items])
        self.index = {int(aid): index for index, aid in enumerate(self.ids)}
        if len(self.index) != len(self.items):
            raise ValueError("Duplicate catalog IDs")
        selection_path = root / "models/release_manifest.json"
        if not selection_path.exists():
            raise FileNotFoundError("models/release_manifest.json is required")
        self.selection = read_json(selection_path)
        self.variant = "optimized"
        model_file = "models/logistic_optimized.json"
        vector_file = "models/bge_optimized.npz"
        metadata_file = "models/bge_optimized_metadata.json"
        self.model_files = [model_file, vector_file, metadata_file]
        self.logistic = read_json(root / model_file)
        metadata = read_json(root / metadata_file)
        if self.selection:
            for name in self.model_files:
                if sha256(root / name) != self.selection["files"][name]:
                    raise ValueError("Selected artifact differs from frozen model selection")
        catalog_hash = sha256(root / "data/catalog.jsonl.gz")
        if self.logistic["catalog_sha256"] != catalog_hash or metadata["catalog_sha256"] != catalog_hash:
            raise ValueError("Model and catalog versions differ")
        if sha256(root / vector_file) != metadata["vector_sha256"]:
            raise ValueError("Embedding artifact hash mismatch")
        with np.load(root / vector_file, allow_pickle=False) as data:
            if not np.array_equal(data["anime_ids"], self.ids):
                raise ValueError("Embedding row mapping differs from catalog")
            self.vectors = data["vectors"].copy()
        if self.vectors.shape != (len(self.items), 384) or not np.all(np.isfinite(self.vectors)):
            raise ValueError("Invalid embedding dimensions or values")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Embeddings must be unit-normalized")

    def scores(self, anime_id):
        position = self.index[anime_id]
        features = feature_matrix(self.items[position], self.items)
        # Keep the feature summation order stable so exact ties remain reproducible.
        tag = np.zeros(len(self.items), dtype=np.float64)
        for column, weight in enumerate(WEIGHTS):
            tag += features[:, column] * weight
        learned = expit(features @ np.asarray(self.logistic["coefficients"]) + self.logistic["intercept"])
        cosine = np.clip(self.vectors @ self.vectors[position], -1.0, 1.0)
        normalization = self.selection["D"]
        calibration = self.selection["semantic_calibration"]
        fusion = 0.6 * tag + 0.4 * normalized_semantic(cosine, normalization, calibration)
        return {"A": tag, "B": learned, "C": cosine, "D": fusion}, features

    def recommend(self, anime_id, method="A", k=5):
        """Same input/output for all methods; never receives or reads user history."""
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        scores, features = self.scores(anime_id)
        ranked = rank_top_k(self.ids, scores[method], anime_id, k)
        return [dict(anime_id=int(self.ids[index]), title=self.items[index]["title"],
                     similarity=float(scores[method][index]),
                     dimensions=dict(zip(DIMENSIONS, features[index].tolist()))) for index in ranked]
