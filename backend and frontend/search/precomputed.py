from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: str
    score: float


class PrecomputedBM25SIndex:
    """Adapter for bm25s cache folders produced by notebooks/classical_retrieval.ipynb."""

    def __init__(
        self,
        directory: Path,
        *,
        stopwords: str | None = "ru",
        stemmer_name: str | None = "russian",
    ) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("Install bm25s to use PRECOMPUTED_BM25S_DIR") from exc

        self.bm25s = bm25s
        self.directory = directory
        self.index_dir = directory / "index" if (directory / "index").exists() else directory
        if not (self.index_dir / "params.index.json").exists():
            raise FileNotFoundError(
                f"BM25S index files not found in {self.index_dir}; expected params.index.json"
            )
        self.index = bm25s.BM25.load(str(self.index_dir))
        self.chunk_ids = _load_chunk_ids(
            directory,
            [
                "chunk_ids.pkl",
                "bm25s_chunk_ids.pkl",
                "bm25_chunk_ids.pkl",
                "index/chunk_ids.pkl",
                "index/bm25s_chunk_ids.pkl",
                "index/bm25_chunk_ids.pkl",
            ],
            required=False,
        )
        self.stopwords = stopwords
        self.stemmer = None
        if stemmer_name:
            try:
                import Stemmer

                self.stemmer = Stemmer.Stemmer(stemmer_name)
            except ImportError as exc:
                raise ImportError("Install PyStemmer to use BM25S stemming") from exc

    def search(self, query: str, *, limit: int = 10) -> list[ChunkHit]:
        if limit < 1:
            return []
        query_tokens = self.bm25s.tokenize(
            [query],
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )
        indices, scores = self.index.retrieve(query_tokens, k=limit)
        return _hits_from_indices(indices[0], scores[0], self.chunk_ids)


class PrecomputedDenseIndex:
    """Adapter for dense/faiss_index.bin + dense_chunk_ids.pkl.

    If faiss is unavailable, falls back to chunk_embeddings.npy with a numpy inner-product scan.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.chunk_ids = _load_chunk_ids(
            directory,
            ["dense_chunk_ids.pkl", "chunk_ids.pkl"],
            required=True,
        )
        self.index = None
        self.embeddings = None
        faiss_path = directory / "faiss_index.bin"
        if faiss_path.exists():
            try:
                import faiss

                self.index = faiss.read_index(str(faiss_path))
            except ImportError:
                self.index = None
        if self.index is None:
            embeddings_path = directory / "chunk_embeddings.npy"
            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"Dense index needs {faiss_path} with faiss installed or {embeddings_path}"
                )
            self.embeddings = np.load(embeddings_path, mmap_mode="r")

    def search(self, query_vector: np.ndarray, *, limit: int = 10) -> list[ChunkHit]:
        if limit < 1:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self.index is not None:
            scores, indices = self.index.search(query_vector, limit)
            return _hits_from_indices(indices[0], scores[0], self.chunk_ids)

        scores = np.asarray(self.embeddings @ query_vector.ravel()).ravel()
        take = min(limit, scores.shape[0])
        if take < scores.shape[0]:
            chosen = np.argpartition(scores, -take)[-take:]
        else:
            chosen = np.arange(scores.shape[0])
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        return [
            ChunkHit(chunk_id=str(self.chunk_ids[index]), score=float(scores[index]))
            for index in chosen[:take]
        ]


class PrecomputedSPLADEIndex:
    """Adapter for splade_index.npz + splade_chunk_ids.npy from notebooks/eval_splade.py."""

    def __init__(self, directory: Path) -> None:
        matrix_path = directory / "splade_index.npz"
        chunk_ids_path = directory / "splade_chunk_ids.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"SPLADE matrix not found: {matrix_path}")
        if not chunk_ids_path.exists():
            raise FileNotFoundError(f"SPLADE chunk ids not found: {chunk_ids_path}")
        self.matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
        self.chunk_ids = np.load(chunk_ids_path, allow_pickle=True)
        if self.matrix.shape[0] != len(self.chunk_ids):
            raise ValueError("SPLADE matrix/chunk_ids row mismatch")

    def search(self, query_vector: sparse.csr_matrix, *, limit: int = 10) -> list[ChunkHit]:
        if limit < 1:
            return []
        query_vector = query_vector.tocsr()
        if query_vector.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                f"SPLADE query has {query_vector.shape[1]} features, index expects "
                f"{self.matrix.shape[1]}"
            )
        scores = (query_vector @ self.matrix.T).toarray().ravel()
        take = min(limit, scores.shape[0])
        if take < scores.shape[0]:
            chosen = np.argpartition(scores, -take)[-take:]
        else:
            chosen = np.arange(scores.shape[0])
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        return [
            ChunkHit(chunk_id=str(self.chunk_ids[index]), score=float(scores[index]))
            for index in chosen[:take]
            if scores[index] > 0
        ]


def _load_chunk_ids(directory: Path, names: list[str], *, required: bool) -> list[str] | None:
    for name in names:
        path = directory / name
        if path.exists():
            if path.suffix == ".npy":
                return [str(value) for value in np.load(path, allow_pickle=True).tolist()]
            with path.open("rb") as handle:
                values = pickle.load(handle)
            return [str(value) for value in values]
    if required:
        raise FileNotFoundError(
            f"Chunk id file not found in {directory}; tried: {', '.join(names)}"
        )
    return None


def _hits_from_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    chunk_ids: list[str] | None,
) -> list[ChunkHit]:
    hits: list[ChunkHit] = []
    for index, score in zip(indices, scores, strict=False):
        index = int(index)
        if index < 0:
            continue
        chunk_id = str(chunk_ids[index]) if chunk_ids is not None else str(index)
        hits.append(ChunkHit(chunk_id=chunk_id, score=float(score)))
    return hits
