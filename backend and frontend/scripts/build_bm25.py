from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.bm25 import BM25Index
from search.config import get_settings
from search.data import DataRepository


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a persistent BM25 index")
    parser.add_argument("--mapping", type=Path, default=settings.data_path)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_path)
    parser.add_argument("--output-dir", type=Path, default=settings.bm25_dir)
    parser.add_argument("--k1", type=float, default=settings.bm25_k1)
    parser.add_argument("--b", type=float, default=settings.bm25_b)
    parser.add_argument("--min-df", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DataRepository(args.mapping, args.corpus)
    content_kind = "title+" + repository.content_column if repository.content_column else "title"
    print(f"Building BM25 over {len(repository):,} rows ({content_kind})...")
    index = BM25Index.build(
        repository.search_texts(),
        repository.frame["source"].tolist(),
        k1=args.k1,
        b=args.b,
        min_df=args.min_df,
        fingerprint=repository.fingerprint(),
        content_kind=content_kind,
    )
    index.save(args.output_dir)
    print(
        f"Saved {index.matrix.shape[0]:,} x {index.matrix.shape[1]:,} BM25 matrix "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
