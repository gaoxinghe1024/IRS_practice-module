"""Text and unlabeled normalization candidates: same five fields, same BGE model."""
import argparse
import hashlib

import numpy as np

from .common import ROOT, read_rows, read_json, sha256, write_json
from .embed import encode_texts, MODEL, REVISION
from .features import semantic_text

CACHE = ROOT / ".cache/representations"


def describe(item, style):
    if style == "original":
        return semantic_text(item)
    if style not in {"compact", "natural"}:
        raise ValueError("Unknown five-field text template")
    if style == "compact":
        fields = [", ".join(item["genres"]), ", ".join(item["themes"]),
                  item["demographic"], item["studio"],
                  f"{item['episodes']} episodes" if item["episodes"] else None]
        return ". ".join(value for value in fields if value) + "." if any(fields) else "Anime."
    sentences = []
    if item["genres"]:
        sentences.append("An anime in the " + ", ".join(item["genres"]) + " genres.")
    if item["themes"]:
        sentences.append("Its themes are " + ", ".join(item["themes"]) + ".")
    if item["demographic"]:
        sentences.append("Its demographic is " + item["demographic"] + ".")
    if item["studio"]:
        sentences.append("Produced by " + item["studio"] + ".")
    if item["episodes"]:
        sentences.append(f"It has {item['episodes']} episodes.")
    return " ".join(sentences) or "Anime."


def normalize_vectors(vectors, centered=False):
    vectors = np.asarray(vectors, dtype=np.float32)
    mean = vectors.mean(axis=0, dtype=np.float64).astype(np.float32) if centered else np.zeros(vectors.shape[1], dtype=np.float32)
    transformed = vectors - mean
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Centering would produce a zero vector")
    return transformed / norms, mean


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
    for style in ("compact", "natural"):
        path = CACHE / f"{style}.npz"
        meta_path = CACHE / f"{style}.json"
        texts = [describe(item, style) for item in items]
        text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        if path.exists() and meta_path.exists():
            meta = read_json(meta_path)
            if meta["text_sha256"] == text_hash and meta["vector_sha256"] == sha256(path):
                print(f"Reuse {style} representation", flush=True)
                continue
        vectors, truncated, seconds = encode_texts(texts, label=style)
        np.savez_compressed(path, anime_ids=np.asarray([row["anime_id"] for row in items]), vectors=vectors)
        write_json(meta_path, dict(model=MODEL, revision=REVISION, text_style=style, dimensions=384,
                                  pooling="CLS", catalog_sha256=sha256(ROOT / "data/catalog.jsonl.gz"),
                                  text_sha256=text_hash, vector_sha256=sha256(path), truncated_titles=truncated,
                                  encode_seconds=seconds, uses_only_five_dimensions=True))
    print("Two additional text representations ready; no truth labels used", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
