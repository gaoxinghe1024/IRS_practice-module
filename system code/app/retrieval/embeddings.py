"""Replaceable text-embedding adapters. All vectors are L2-normalised float32.

bge  BAAI/bge-small-en-v1.5 via sentence-transformers (optional ``ml`` extra).
     English model, 384 dimensions, 512-token input limit (model card). It is not
     assumed to work for Chinese / cross-lingual queries.
lsa  TF-IDF + truncated SVD fitted on the corpus (scikit-learn only). A lexical-
     semantic approximation used when the ml extra is missing; reported under its
     own name, never as BGE.

Cosine scores from either adapter are similarities, not probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    provider: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def info(self) -> dict: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _hf_revision(model_name: str) -> str | None:
    import os

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = cache / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


class BGEEncoder:
    provider = "bge"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64, device: str | None = None,
                 local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device or "cpu", local_files_only=local_files_only)
        self.max_seq_length = int(self.model.max_seq_length)
        self.tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                            show_progress_bar=False, convert_to_numpy=True))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents([BGE_QUERY_INSTRUCTION + t for t in texts])

    @property
    def dim(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model_name, "revision": _hf_revision(self.model_name),
                "dim": self.dim, "max_seq_length": self.max_seq_length,
                "query_instruction": BGE_QUERY_INSTRUCTION}


class LSAEncoder:
    provider = "lsa"

    def __init__(self, dim: int = 256):
        self.dim_target = dim
        self.vectorizer = None
        self.svd = None

    def fit(self, texts: list[str]) -> LSAEncoder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200000,
                                          stop_words="english", strip_accents="unicode")
        X = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dim_target, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.svd.fit(X)
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("LSA encoder is not fitted")
        return _normalise(self.svd.transform(self.vectorizer.transform(texts)))

    encode_documents = _encode
    encode_queries = _encode

    @property
    def dim(self) -> int:
        return int(self.svd.n_components) if self.svd is not None else self.dim_target

    def count_tokens(self, text: str) -> int:
        from app.core.text import approx_tokens

        return approx_tokens(text)

    def info(self) -> dict:
        return {"provider": self.provider, "model": "tfidf-bigram+truncated-svd", "revision": None,
                "dim": self.dim, "note": "lexical-semantic approximation (no neural model)"}

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim_target}, path)

    @classmethod
    def load(cls, path: Path) -> LSAEncoder:
        import joblib

        blob = joblib.load(path)
        enc = cls(blob["dim"])
        enc.vectorizer, enc.svd = blob["vectorizer"], blob["svd"]
        return enc


def choose_provider(cfg: dict) -> str:
    provider = (cfg or {}).get("provider", "auto")
    if provider == "auto":
        return "bge" if sentence_transformers_available() else "lsa"
    return provider
