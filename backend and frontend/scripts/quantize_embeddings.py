from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a symmetric int8 embedding artefact")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--validation-sample", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectors = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {vectors.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized_path = args.output.with_suffix(".int8.npy")
    scales_path = args.output.with_suffix(".scales.npy")
    quantized = np.lib.format.open_memmap(
        quantized_path, mode="w+", dtype=np.int8, shape=vectors.shape
    )
    scales = np.lib.format.open_memmap(
        scales_path, mode="w+", dtype=np.float32, shape=(len(vectors), 1)
    )
    for start in range(0, len(vectors), args.batch_size):
        end = min(len(vectors), start + args.batch_size)
        batch = np.asarray(vectors[start:end], dtype=np.float32)
        max_abs = np.max(np.abs(batch), axis=1, keepdims=True)
        scale = np.maximum(max_abs / 127.0, np.finfo(np.float32).eps)
        quantized[start:end] = np.clip(np.rint(batch / scale), -127, 127).astype(np.int8)
        scales[start:end] = scale
    quantized.flush()
    scales.flush()

    sample_size = min(args.validation_sample, len(vectors))
    if sample_size:
        rng = np.random.default_rng(42)
        ids = rng.choice(len(vectors), sample_size, replace=False)
        original = np.asarray(vectors[ids], dtype=np.float32)
        restored = np.asarray(quantized[ids], dtype=np.float32) * np.asarray(scales[ids])
        dot = np.sum(original * restored, axis=1)
        denom = np.linalg.norm(original, axis=1) * np.linalg.norm(restored, axis=1)
        cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom != 0)
        cosine_mean = float(cosine.mean())
    else:
        cosine_mean = 1.0
    original_bytes = int(args.input.stat().st_size)
    quantized_bytes = int(quantized_path.stat().st_size + scales_path.stat().st_size)
    report = {
        "rows": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "scheme": "symmetric_per_vector_int8",
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": original_bytes / quantized_bytes,
        "sample_mean_cosine_original_vs_dequantized": cosine_mean,
        "note": "Qdrant uses its own scalar quantization; this pair is an offline analysis artefact.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

