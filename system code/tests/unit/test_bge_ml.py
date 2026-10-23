"""BGE encoder checks (marker `ml`): needs the ml extra and the cached model; skipped otherwise."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def encoder():
    pytest.importorskip("sentence_transformers")
    from app.retrieval.embeddings import BGEEncoder

    try:
        return BGEEncoder("BAAI/bge-small-en-v1.5", batch_size=8, device="cpu")
    except Exception as exc:  # noqa: BLE001 - model not downloaded / offline
        pytest.skip(f"BGE model unavailable: {type(exc).__name__}")


def test_bge_vectors_are_384d_normalised_and_meaningful(encoder):
    v = np.asarray(encoder.encode_documents(["a detective hunts a serial killer", "an investigator tracks a murderer",
                                   "a cooking contest at a high school"]))
    assert v.shape == (3, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert v[0] @ v[1] > v[0] @ v[2]                      # paraphrase closer than an unrelated sentence


def test_bge_token_counter_and_chunks_respect_the_512_limit(encoder):
    from app.retrieval.documents import chunk_text

    text = " ".join(f"Sentence {i} tells a small part of a long story about two travellers." for i in range(400))
    assert encoder.count_tokens(text) > 512
    chunks = chunk_text(text, encoder.count_tokens, 150, 350, 40)
    assert chunks and max(encoder.count_tokens(c[2]) for c in chunks) <= 512
