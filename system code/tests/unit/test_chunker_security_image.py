import io

import numpy as np
import pytest

from app.core.errors import ValidationFailed
from app.core.security import LEGACY_PREFIX, hash_password, verify_password
from app.core.text import approx_tokens
from app.multimodal.image import ColorHistogramEncoder, ImageGallery, validate_upload
from app.retrieval.documents import build_documents, chunk_text


def test_chunker_respects_limits_and_keeps_short_text_whole():
    short = "One sentence only."
    assert chunk_text(short, approx_tokens, 150, 350, 40) == [(0, len(short), short)]
    long = " ".join(f"Sentence number {i} describes an event in the story." for i in range(200))
    chunks = chunk_text(long, approx_tokens, 150, 350, 40)
    assert len(chunks) > 3
    assert all(approx_tokens(c[2]) <= 350 for c in chunks)
    starts = [c[0] for c in chunks]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)        # always advances
    assert chunks[0][0] == 0 and chunks[-1][1] == len(long)                     # full coverage


def test_long_relation_lists_are_packed_into_bounded_chunks_of_whole_facts():
    import pandas as pd

    keys = [f"synthetic:1:{i:03d}" for i in range(120)]
    catalog = pd.DataFrame({"item_key": keys, "title": [f"Long Franchise Title {i}" for i in range(120)],
                            "in_catalog": True, "synopsis": None})
    rel = pd.DataFrame({"from_item": keys[0], "to_item": keys[1:], "to_entity_type": "anime", "to_name": None,
                        "relation_type": "side_story", "source_ref": "fixture"})
    empty = pd.DataFrame(columns=["item_key", "namespace", "value"])
    docs = [d for d in build_documents(catalog, empty, rel, pd.DataFrame(), None, None, approx_tokens,
                                       {"chunk_max_tokens": 120}) if d.kind == "relations"]
    assert len(docs) > 1
    assert all(approx_tokens(d.body) <= 120 for d in docs)
    assert [d.doc_id for d in docs] == [f"{keys[0]}#rel-{n}" for n in range(len(docs))]
    facts = [f for d in docs for f in d.body.split(". ") if f]
    assert len(facts) == 119 and all(f.startswith("Long Franchise Title 0 has side story") for f in facts)


def test_password_hashing_and_legacy_verification():
    h = hash_password("s3cret-pass")
    assert verify_password("s3cret-pass", h) == (True, False)
    assert verify_password("wrong", h)[0] is False
    import hashlib

    legacy = LEGACY_PREFIX + "abcd$" + hashlib.pbkdf2_hmac("sha256", b"123456", b"abcd", 60000).hex()
    assert verify_password("123456", legacy) == (True, True)        # ok, and must be re-hashed
    assert verify_password("1234567", legacy)[0] is False


def _png(color=(10, 200, 30), size=(32, 32)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_validation():
    img, info = validate_upload(_png(), 5_000_000, 10_000_000)
    assert info["format"] == "PNG" and img.size == (32, 32)
    with pytest.raises(ValidationFailed) as e:
        validate_upload(b"not an image", 5_000_000, 10_000_000)
    assert e.value.code == "image_corrupt"
    with pytest.raises(ValidationFailed) as e:
        validate_upload(_png(), 10, 10_000_000)
    assert e.value.code == "image_too_large"
    with pytest.raises(ValidationFailed) as e:
        validate_upload(_png(size=(200, 200)), 5_000_000, 1000)
    assert e.value.code == "image_too_many_pixels"
    with pytest.raises(ValidationFailed):
        validate_upload(b"", 5_000_000, 10_000_000)


def test_gallery_identify_is_uncalibrated_by_default():
    from PIL import Image

    enc = ColorHistogramEncoder()
    imgs = [Image.new("RGB", (32, 32), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    vecs = np.vstack([enc.encode_image(i) for i in imgs])
    meta = [{"image_id": f"i{n}", "item_key": f"synthetic:1:00{n}", "scene_id": None, "episode": None,
             "timestamp_seconds": None} for n in range(3)]
    res = ImageGallery(vecs, meta, enc, "test").identify(Image.new("RGB", (32, 32), (0, 250, 5)))
    assert res["candidates"][0]["item_key"] == "synthetic:1:001"
    assert res["accepted"] is None and res["calibrated"] is False
    assert res["candidates"][0]["frames"][0]["episode"] is None     # unknown stays unknown
