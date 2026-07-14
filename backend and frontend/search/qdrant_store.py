from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .data import DataRepository


@dataclass(frozen=True, slots=True)
class DenseHit:
    point_id: int | str
    row_index: int
    doc_id: str
    score: float
    payload: dict


def batched_range(length: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, length, batch_size):
        yield start, min(length, start + batch_size)


class QdrantStore:
    def __init__(self, url: str, api_key: str | None = None, timeout: int = 60) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to use dense search") from exc
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif url.startswith("file://"):
            self.client = QdrantClient(path=url.removeprefix("file://"))
        else:
            self.client = QdrantClient(
                url=url, api_key=api_key or None, timeout=timeout, trust_env=False
            )

    @staticmethod
    def _models():
        from qdrant_client import models

        return models

    def collection_exists(self, name: str) -> bool:
        return bool(self.client.collection_exists(name))

    def collection_dimension(self, name: str) -> int:
        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            vector_params = next(iter(vectors.values()))
            return int(vector_params.size)
        return int(vectors.size)

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        recreate: bool = False,
        hnsw: bool = True,
        quantized: bool = False,
        hnsw_m: int = 32,
        ef_construct: int = 200,
    ) -> None:
        models = self._models()
        if self.collection_exists(name):
            if not recreate:
                raise RuntimeError(f"Collection {name!r} already exists; pass --recreate")
            self.client.delete_collection(name)

        quantization = None
        if quantized:
            quantization = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            )
        hnsw_config = models.HnswConfigDiff(
            m=hnsw_m if hnsw else 0,
            ef_construct=ef_construct,
            full_scan_threshold=10_000,
        )
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT32,
            ),
            hnsw_config=hnsw_config,
            quantization_config=quantization,
        )
        self.client.create_payload_index(
            collection_name=name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        self.client.create_payload_index(
            collection_name=name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def upload(
        self,
        name: str,
        embeddings: np.ndarray,
        repository: DataRepository,
        *,
        batch_size: int = 512,
    ) -> None:
        models = self._models()
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if len(embeddings) != len(repository):
            raise ValueError(
                f"Embedding rows ({len(embeddings)}) != mapping rows ({len(repository)})"
            )
        expected_dim = self.collection_dimension(name)
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != collection dimension {expected_dim}"
            )

        for start, end in batched_range(len(repository), batch_size):
            vector_batch = np.asarray(embeddings[start:end], dtype=np.float32)
            if not np.isfinite(vector_batch).all():
                raise ValueError(f"Non-finite embedding value in rows {start}:{end}")
            points = [
                models.PointStruct(
                    id=row_index,
                    vector=vector_batch[row_index - start].tolist(),
                    payload=repository.payload(row_index),
                )
                for row_index in range(start, end)
            ]
            self.client.upsert(collection_name=name, points=points, wait=True)

        count = self.client.count(collection_name=name, exact=True).count
        if count != len(repository):
            raise RuntimeError(f"Qdrant contains {count} points, expected {len(repository)}")

    def search(
        self,
        name: str,
        vector: np.ndarray,
        *,
        limit: int,
        source_filter: set[str] | None = None,
        exact: bool = False,
        hnsw_ef: int = 128,
    ) -> list[DenseHit]:
        models = self._models()
        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchAny(any=sorted(source_filter))
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=name,
            query=np.asarray(vector, dtype=np.float32).ravel().tolist(),
            query_filter=query_filter,
            search_params=models.SearchParams(exact=exact, hnsw_ef=None if exact else hnsw_ef),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[DenseHit] = []
        for point in response.points:
            payload = point.payload or {}
            row_index = int(payload.get("row_index", point.id))
            hits.append(
                DenseHit(
                    point_id=point.id,
                    row_index=row_index,
                    doc_id=str(payload.get("doc_id", "")),
                    score=float(point.score),
                    payload=payload,
                )
            )
        return hits

    def collection_stats(self, name: str) -> dict[str, int | str]:
        info = self.client.get_collection(name)
        return {
            "status": str(info.status),
            "points_count": int(info.points_count or 0),
            "indexed_vectors_count": int(info.indexed_vectors_count or 0),
            "segments_count": int(info.segments_count or 0),
            "dimension": self.collection_dimension(name),
        }

    def wait_until_indexed(
        self,
        name: str,
        *,
        expected_vectors: int,
        timeout_seconds: int = 7200,
        poll_seconds: float = 5.0,
    ) -> dict[str, int | str]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            stats = self.collection_stats(name)
            if int(stats["indexed_vectors_count"]) >= expected_vectors:
                return stats
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Qdrant indexed {stats['indexed_vectors_count']}/{expected_vectors} vectors "
                    f"within {timeout_seconds}s"
                )
            time.sleep(poll_seconds)
