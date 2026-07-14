from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.data import DataRepository
from search.qdrant_store import QdrantStore


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Load an embedding matrix into Qdrant")
    parser.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    parser.add_argument("--mapping", type=Path, default=settings.data_path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--collection")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construct", type=int, default=200)
    parser.add_argument("--index-timeout", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    embeddings_path = args.embeddings or (
        settings.embeddings_v1_path if args.iteration == 1 else settings.embeddings_v2_path
    )
    collection = args.collection or (
        settings.qdrant_collection_v1 if args.iteration == 1 else settings.qdrant_collection_v2
    )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    repository = DataRepository(args.mapping)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(repository):
        raise ValueError(
            f"Expected ({len(repository)}, dimension), got {embeddings.shape}. "
            "Do not reorder/deduplicate the mapping."
        )
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, timeout=120)
    store.create_collection(
        collection,
        int(embeddings.shape[1]),
        recreate=args.recreate,
        hnsw=args.iteration == 2,
        quantized=args.iteration == 2,
        hnsw_m=args.hnsw_m,
        ef_construct=args.ef_construct,
    )
    store.upload(collection, embeddings, repository, batch_size=args.batch_size)
    if args.iteration == 2:
        print("Waiting for the HNSW optimizer to index all vectors...")
        store.wait_until_indexed(
            collection,
            expected_vectors=len(repository),
            timeout_seconds=args.index_timeout,
        )
    print(store.collection_stats(collection))


if __name__ == "__main__":
    main()
