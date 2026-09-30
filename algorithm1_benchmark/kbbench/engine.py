"""Common recommendation interface for the four replaceable similarity modules."""
import numpy as np
from scipy.special import expit

from .common import ROOT, read_json, read_rows, sha256
from .features import DIMENSIONS, WEIGHTS, cosine_to_unit, feature_matrix, rank_top_k

METHODS = {"A": "Weighted five-dimension similarity", "B": "Logistic regression",
           "C": "BGE embeddings", "D": "0.6 tag + 0.4 semantic"}


class SimilarityEngine:
    def __init__(self, root=ROOT):
        self.items = list(read_rows(root / "data/catalog.jsonl.gz"))
        self.ids = np.asarray([item["anime_id"] for item in self.items])
        self.index = {int(aid): index for index, aid in enumerate(self.ids)}
        if len(self.index) != len(self.items):
            raise ValueError("Duplicate catalog IDs")
        self.logistic = read_json(root / "models/logistic.json")
        metadata = read_json(root / "models/bge_metadata.json")
        catalog_hash = sha256(root / "data/catalog.jsonl.gz")
        if self.logistic["catalog_sha256"] != catalog_hash or metadata["catalog_sha256"] != catalog_hash:
            raise ValueError("Model and catalog versions differ")
        if sha256(root / "models/bge_embeddings.npz") != metadata["vector_sha256"]:
            raise ValueError("Embedding artifact hash mismatch")
        with np.load(root / "models/bge_embeddings.npz", allow_pickle=False) as data:
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
        # Match the standard-library baseline's summation order, including exact ties.
        tag = np.zeros(len(self.items), dtype=np.float64)
        for column, weight in enumerate(WEIGHTS):
            tag += features[:, column] * weight
        learned = expit(features @ np.asarray(self.logistic["coefficients"]) + self.logistic["intercept"])
        cosine = np.clip(self.vectors @ self.vectors[position], -1.0, 1.0)
        fusion = 0.6 * tag + 0.4 * cosine_to_unit(cosine)
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
