from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.data import DataRepository
from search.splade import SPLADEIndex, create_sparse_encoder


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a persisted SPLADE sparse index")
    parser.add_argument("--data", type=Path, default=settings.data_path)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_path)
    parser.add_argument("--output", type=Path, default=settings.splade_dir)
    parser.add_argument("--model", default=settings.splade_model)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--batch-size", type=int, default=settings.splade_batch_size)
    parser.add_argument("--max-length", type=int, default=settings.splade_max_length)
    parser.add_argument("--threshold", type=float, default=settings.splade_threshold)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=settings.splade_fp16)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit; do not use for final metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DataRepository(args.data, args.corpus)
    texts = repository.search_texts()
    sources = repository.frame["source"].astype(str).to_numpy()
    row_numbers = np.arange(len(texts), dtype=np.int64)
    fingerprint = repository.fingerprint()

    if args.limit:
        texts = texts[: args.limit]
        sources = sources[: args.limit]
        row_numbers = row_numbers[: args.limit]
        fingerprint = f"{fingerprint}:limit={args.limit}"

    encoder = create_sparse_encoder(
        args.model,
        device=args.device,
        max_length=args.max_length,
        threshold=args.threshold,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    matrix, order = encoder.encode(
        texts,
        batch_size=args.batch_size,
        sort_by_length=True,
        show_progress=True,
    )

    index = SPLADEIndex(
        matrix=matrix,
        row_indices=row_numbers[order],
        sources=sources[order],
        metadata={
            "version": 1,
            "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "model": args.model,
            "max_length": args.max_length,
            "threshold": args.threshold,
            "fingerprint": fingerprint,
            "content_kind": repository.content_column or "title",
            "ordering": "matrix row i maps to row_indices[i]; queries are encoded unsorted",
        },
    )
    index.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": matrix.shape[0],
                "features": matrix.shape[1],
                "nnz": int(matrix.nnz),
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
