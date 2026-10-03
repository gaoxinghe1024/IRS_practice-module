"""BGE small English v1.5: pinned original ONNX weights, CLS pooling, unit L2 vectors."""
import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .common import ROOT, read_json, read_rows, sha256, utc_now, write_json
from .features import semantic_text
from .network import download_file

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
FILES = ["onnx/model.onnx", "tokenizer.json", "config.json", "1_Pooling/config.json", "README.md"]
MODEL_DIR = ROOT / ".cache/bge-small-en-v1.5"
EXPECTED_ONNX_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
EXPECTED_TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"


def download():
    for name in FILES:
        path = MODEL_DIR / name
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true"
        temp = path.with_suffix(path.suffix + ".part")
        download_file(url, temp, timeout=300)
        os.replace(temp, path)
        print(f"Downloaded {name}: {path.stat().st_size:,} bytes", flush=True)
    write_json(MODEL_DIR / "download_manifest.json", dict(model=MODEL, revision=REVISION, retrieved_at=utc_now(),
               files={name: dict(sha256=sha256(MODEL_DIR / name), bytes=(MODEL_DIR / name).stat().st_size) for name in FILES}))


def encode_texts(texts, batch_size=64, threads=4, label="BGE"):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    started = time.perf_counter()
    if batch_size < 1 or threads < 1:
        raise ValueError("batch-size and threads must be positive")
    if sha256(MODEL_DIR / "onnx/model.onnx") != EXPECTED_ONNX_SHA256 or sha256(MODEL_DIR / "tokenizer.json") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Model/tokenizer hash differs from the pinned release")
    pooling = read_json(MODEL_DIR / "1_Pooling/config.json")
    if not pooling.get("pooling_mode_cls_token") or pooling.get("pooling_mode_mean_tokens"):
        raise ValueError("Model pooling config is not CLS-only")
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL_DIR / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    result = np.empty((len(texts), 384), dtype=np.float32)
    truncated = 0
    input_names = {entry.name for entry in session.get_inputs()}
    for start in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(texts[start:start + batch_size])
        truncated += sum(bool(encoding.overflowing) for encoding in encodings)
        inputs = dict(input_ids=np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
                      attention_mask=np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64),
                      token_type_ids=np.asarray([encoding.type_ids for encoding in encodings], dtype=np.int64))
        outputs = session.run(None, {key: value for key, value in inputs.items() if key in input_names})
        hidden = next((output for output in outputs if output.ndim == 3 and output.shape[-1] == 384), None)
        if hidden is None:
            raise ValueError(f"Expected token-level 384-dim ONNX output, received {[x.shape for x in outputs]}")
        vectors = hidden[:, 0, :]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        result[start:start + len(encodings)] = vectors
        if start % (batch_size * 10) == 0:
            print(f"{label} {start + len(encodings)}/{len(texts)}, elapsed {time.perf_counter()-started:.1f}s", flush=True)
    if not np.all(np.isfinite(result)) or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid embeddings")
    return result, truncated, time.perf_counter() - started


def encode(batch_size=64, threads=4):
    items = list(read_rows(ROOT / "data/catalog.jsonl.gz"))
    texts = [semantic_text(item) for item in items]
    result, truncated, seconds = encode_texts(texts, batch_size, threads)
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "models/bge_optimized.npz", anime_ids=np.array([row["anime_id"] for row in items]), vectors=result)
    metadata = dict(model=MODEL, revision=REVISION, backend="original ONNX float32; CPUExecutionProvider",
                    dimensions=384, pooling="CLS", l2_normalized=True, max_length=128,
                    batch_size=batch_size, threads=threads, instruction_prefix=None,
                    justification="Symmetric item-to-item similarity; five attributes only; no retrieval instruction prefix",
                    catalog_sha256=sha256(ROOT / "data/catalog.jsonl.gz"),
                    text_sha256=hashlib.sha256("\n".join(texts).encode()).hexdigest(),
                    vector_sha256=sha256(ROOT / "models/bge_optimized.npz"),
                    onnx_sha256=sha256(MODEL_DIR / "onnx/model.onnx"),
                    tokenizer_sha256=sha256(MODEL_DIR / "tokenizer.json"),
                    titles=len(items), truncated_titles=truncated, encode_seconds=seconds)
    write_json(ROOT / "models/bge_metadata.json", metadata)
    print(metadata, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.download or args.download_only:
        download()
    if not args.download_only:
        encode(args.batch_size, args.threads)
