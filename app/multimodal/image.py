"""Screenshot identification pipeline (data-conditional extension).

Real identification needs frame-level images (or a reliable external service);
cover art is NOT a substitute and is never used to claim screenshot recognition.

encoders
  color_histogram_v1  deterministic colour/spatial histogram (Pillow + numpy). Used to validate the
                      pipeline on synthetic fixture frames only; never used for accuracy claims.
  clip                sentence-transformers CLIP (optional ml extra); dimension read from the model.

Uploads are decoded and validated (size, pixel count, format) before any use,
and are processed in memory unless the user explicitly saves them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.errors import ValidationFailed

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def validate_upload(data: bytes, max_bytes: int, max_pixels: int):
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ValidationFailed("The uploaded file is empty.", code="image_empty")
    if len(data) > max_bytes:
        raise ValidationFailed(f"Image is larger than {max_bytes // (1024 * 1024)} MiB.", code="image_too_large")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ValidationFailed(f"Unsupported image format {fmt}. Use PNG, JPEG or WebP.",
                                       code="image_unsupported_format")
            if w * h > max_pixels:
                raise ValidationFailed("Image has too many pixels.", code="image_too_many_pixels")
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except ValidationFailed:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValidationFailed("The file could not be decoded as an image.", code="image_corrupt") from exc
    return img.convert("RGB"), {"format": fmt, "width": w, "height": h, "bytes": len(data)}


class ColorHistogramEncoder:
    encoder_id = "color_histogram_v1"

    def encode_image(self, img) -> np.ndarray:
        arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.int64)
        feats = [np.bincount(((arr // 32) * np.array([64, 8, 1])).sum(-1).ravel(), minlength=512).astype(float)]
        for y in (0, 32):
            for x in (0, 32):
                q = arr[y:y + 32, x:x + 32] // 64
                feats.append(np.bincount((q * np.array([16, 4, 1])).sum(-1).ravel(), minlength=64).astype(float))
        v = np.concatenate([f / max(1.0, f.sum()) for f in feats])
        return (v / max(1e-12, np.linalg.norm(v))).astype(np.float32)

    def encode_path(self, path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as img:
            return self.encode_image(img)

    def info(self) -> dict:
        return {"encoder": self.encoder_id, "dim": 768, "note": "fixture pipeline validation only"}


class ClipEncoder:
    encoder_id = "clip"

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_image(self, img) -> np.ndarray:
        v = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v.astype(np.float32)

    def encode_path(self, path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as img:
            return self.encode_image(img.convert("RGB"))

    def info(self) -> dict:
        dim = int(self.encode_image(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (8, 8))).shape[0])
        return {"encoder": self.encoder_id, "model": self.model_name, "dim": dim}


def get_image_encoder(cfg: dict):
    name = (cfg or {}).get("encoder", "color_histogram_v1")
    if name == "clip":
        return ClipEncoder((cfg or {}).get("clip_model", "clip-ViT-B-32"))
    return ColorHistogramEncoder()


@dataclass
class ImageMatch:
    item_key: str
    similarity: float
    frames: list[dict]


class ImageGallery:
    def __init__(self, vectors: np.ndarray, meta: list[dict], encoder, source: str,
                 accept_threshold: float | None = None):
        self.vectors = vectors
        self.meta = meta
        self.encoder = encoder
        self.source = source
        self.accept_threshold = accept_threshold

    def identify(self, img, top_n: int = 20, top_items: int = 5) -> dict:
        q = self.encoder.encode_image(img)
        sims = self.vectors @ q
        order = np.argsort(-sims, kind="stable")[:top_n]
        by_item: dict[str, ImageMatch] = {}
        for i in order:
            m = self.meta[int(i)]
            frame = {"image_id": m["image_id"], "scene_id": m.get("scene_id"), "similarity": round(float(sims[i]), 4),
                     "episode": _opt_int(m.get("episode")), "timestamp_seconds": _opt_int(m.get("timestamp_seconds"))}
            cur = by_item.get(m["item_key"])
            if cur is None:
                by_item[m["item_key"]] = ImageMatch(m["item_key"], float(sims[i]), [frame])
            else:
                cur.frames.append(frame)
        ranked = sorted(by_item.values(), key=lambda x: (-x.similarity, x.item_key))[:top_items]
        accepted = None
        if self.accept_threshold is not None and ranked:
            accepted = ranked[0].similarity >= self.accept_threshold
        return {"candidates": [{"item_key": r.item_key, "similarity": round(r.similarity, 4), "frames": r.frames[:3]}
                               for r in ranked],
                "accepted": accepted,
                "calibrated": self.accept_threshold is not None,
                "gallery_source": self.source,
                "note": ("Similarity scores are not probabilities; no acceptance threshold has been "
                         "calibrated, so results are unverified candidates.") if self.accept_threshold is None
                else "Acceptance threshold calibrated on validation data."}


def _opt_int(v):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
