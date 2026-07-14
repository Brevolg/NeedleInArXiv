from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.data import DataRepository
from search.encoders import create_encoder


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Encode mapping/corpus rows into a .npy matrix")
    parser.add_argument("--mapping", type=Path, default=settings.data_path)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_path)
    parser.add_argument("--model", default=settings.model_v1)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DataRepository(args.mapping, args.corpus)
    texts = repository.search_texts()
    encoder = create_encoder(args.model, args.device, args.trust_remote_code)
    if not texts:
        raise ValueError("No documents to encode")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    first_end = min(args.batch_size, len(texts))
    first = encoder.encode(texts[:first_end])
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float32, shape=(len(texts), first.shape[1])
    )
    output[:first_end] = first
    for start in range(first_end, len(texts), args.batch_size):
        end = min(len(texts), start + args.batch_size)
        output[start:end] = encoder.encode(texts[start:end])
        if (start // args.batch_size) % 20 == 0:
            output.flush()
            print(f"Encoded {end:,}/{len(texts):,}")
    output.flush()
    elapsed = time.perf_counter() - started
    sidecar = {
        "model": args.model,
        "rows": len(texts),
        "dimension": int(first.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "mapping_fingerprint": repository.fingerprint(),
        "content": repository.content_column or "title_only",
        "elapsed_seconds": elapsed,
        "documents_per_second": len(texts) / elapsed,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), "utf-8"
    )
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
