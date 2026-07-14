from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings


def main() -> None:
    settings = get_settings()
    errors: list[str] = []

    bm25_dir = settings.precomputed_bm25s_dir
    bm25_index_dir = bm25_dir / "index" if (bm25_dir / "index").exists() else bm25_dir
    for name in (
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "vocab.index.json",
        "params.index.json",
    ):
        if not (bm25_index_dir / name).exists():
            errors.append(f"missing BM25S file: {bm25_index_dir / name}")
    bm25_chunk_ids = first_existing(
        bm25_dir,
        ["chunk_ids.pkl", "bm25s_chunk_ids.pkl", "bm25_chunk_ids.pkl"],
    )
    if bm25_chunk_ids is None:
        errors.append(
            f"missing BM25S chunk ids in {bm25_dir}: chunk_ids.pkl or bm25s_chunk_ids.pkl"
        )

    dense_dir = settings.precomputed_dense_dir
    dense_chunk_ids = dense_dir / "dense_chunk_ids.pkl"
    if not dense_chunk_ids.exists():
        errors.append(f"missing dense chunk ids: {dense_chunk_ids}")
    if not (dense_dir / "faiss_index.bin").exists() and not (dense_dir / "chunk_embeddings.npy").exists():
        errors.append(f"missing dense index: {dense_dir / 'faiss_index.bin'} or chunk_embeddings.npy")
    if dense_chunk_ids.exists() and (dense_dir / "chunk_embeddings.npy").exists():
        ids = load_pickle_len(dense_chunk_ids)
        rows = int(np.load(dense_dir / "chunk_embeddings.npy", mmap_mode="r").shape[0])
        if ids != rows:
            errors.append(f"dense row mismatch: {ids} chunk ids vs {rows} embeddings")

    splade_dir = settings.precomputed_splade_dir
    splade_matrix_path = splade_dir / "splade_index.npz"
    splade_chunk_ids_path = splade_dir / "splade_chunk_ids.npy"
    if not splade_matrix_path.exists():
        errors.append(f"missing SPLADE matrix: {splade_matrix_path}")
    if not splade_chunk_ids_path.exists():
        errors.append(f"missing SPLADE chunk ids: {splade_chunk_ids_path}")
    if splade_matrix_path.exists() and splade_chunk_ids_path.exists():
        rows = sparse.load_npz(splade_matrix_path).shape[0]
        ids = len(np.load(splade_chunk_ids_path, allow_pickle=True))
        if ids != rows:
            errors.append(f"SPLADE row mismatch: {ids} chunk ids vs {rows} matrix rows")

    if settings.chunks_path is None:
        errors.append("CHUNKS_PATH is not set; chunk-level indexes need chunk_id -> doc_id mapping")
    elif not settings.chunks_path.exists():
        errors.append(f"missing chunks parquet: {settings.chunks_path}")

    if errors:
        print("Precomputed index validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Precomputed indexes look consistent.")


def first_existing(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def load_pickle_len(path: Path) -> int:
    with path.open("rb") as handle:
        return len(pickle.load(handle))


if __name__ == "__main__":
    main()
