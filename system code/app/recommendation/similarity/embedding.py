"""Algorithm 1 (C) BGE attribute-text embeddings and (D) tag x semantic hybrid.

C  The same five attributes are rendered with a fixed, versioned template;
   missing attributes are omitted (never invented, and never rendered as a
   shared "unknown" token that would make two sparse titles look alike). Vectors
   are L2-normalised and compared by cosine. Cosine is not a probability.

D  Sim_hybrid = 0.6 * Sim_tag + 0.4 * clip((cos + 1) / 2, 0, 1). When either
   side lacks an embedding, the weight is redistributed to the tag part and the
   degradation is recorded.

A fair first comparison uses the same attribute inputs for A-D. Synopsis-based
similarity is a separate variant (semantic_synopsis), because it changes the
information available.
"""

from __future__ import annotations

import numpy as np

from app.recommendation.features import CatalogSnapshot
from app.recommendation.similarity.weighted import SimilarityResult, WeightedAttributeScorer

TEMPLATE_VERSION = "attr_text_v1"


def attribute_text(snap: CatalogSnapshot, i: int) -> str | None:
    parts = []
    for ns, label in (("genre", "Genres"), ("theme", "Themes"), ("demographic", "Demographic"),
                      ("studio", "Studios")):
        tm = snap.tags.get(ns)
        vals = tm.tags_of(i) if tm is not None else []
        if vals:
            parts.append(f"{label}: {', '.join(vals)}.")
    if np.isfinite(snap.episodes[i]):
        parts.append(f"Episodes: {int(snap.episodes[i])}.")
    return " ".join(parts) if parts else None


class EmbeddingSimilarityScorer:
    scorer_id = "bge_attribute_embedding"
    version = "1.0"

    def __init__(self, vectors: np.ndarray, has_vector: np.ndarray, base: WeightedAttributeScorer, info: dict):
        self.vectors = vectors
        self.has_vector = has_vector
        self.base = base
        self.info = info
        if info.get("provider") != "bge":
            self.scorer_id = f"{info.get('provider', 'unknown')}_attribute_embedding"

    def cosine(self, target: int, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ok = self.has_vector[candidates] & bool(self.has_vector[target])
        cos = np.full(len(candidates), np.nan)
        if ok.any():
            cos[ok] = self.vectors[candidates[ok]] @ self.vectors[target]
        return cos, ok

    def score(self, snap: CatalogSnapshot, target: int, candidates: np.ndarray) -> SimilarityResult:
        res = self.base.score(snap, target, candidates)   # for coverage / explanations only
        cos, ok = self.cosine(target, candidates)
        return SimilarityResult(res.candidates, np.where(ok, cos, np.nan), res.coverage, res.sims, res.available)


class HybridSimilarityScorer:
    scorer_id = "tag_semantic_hybrid"
    version = "1.0"

    def __init__(self, embed: EmbeddingSimilarityScorer, base: WeightedAttributeScorer, tag_weight: float = 0.6):
        self.embed = embed
        self.base = base
        self.tag_weight = tag_weight

    def score(self, snap: CatalogSnapshot, target: int, candidates: np.ndarray) -> SimilarityResult:
        res = self.base.score(snap, target, candidates)
        cos, ok = self.embed.cosine(target, candidates)
        sem = np.clip((np.nan_to_num(cos) + 1.0) / 2.0, 0.0, 1.0)
        tag = res.score
        both = ok & np.isfinite(tag)
        out = np.where(np.isfinite(tag), tag, np.nan)       # degraded: tag only
        out[both] = self.tag_weight * tag[both] + (1 - self.tag_weight) * sem[both]
        only_sem = ok & ~np.isfinite(tag)
        out[only_sem] = sem[only_sem]
        res.sims["semantic"] = np.where(ok, sem, np.nan)
        res.available["semantic"] = ok
        return SimilarityResult(res.candidates, out, res.coverage, res.sims, res.available)
