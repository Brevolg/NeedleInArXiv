from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np

from .bm25 import BM25Index
from .config import Settings
from .data import DataRepository
from .encoders import TextEncoder, create_encoder
from .fusion import RankedItem, reciprocal_rank_fusion
from .qdrant_store import QdrantStore


class SearchMode(StrEnum):
    DENSE_V1 = "dense_v1"
    DENSE_V2 = "dense_v2"
    BM25 = "bm25"
    HYBRID_V1 = "hybrid_v1"
    HYBRID_V2 = "hybrid_v2"


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    row_index: int
    doc_id: str
    title: str
    source: str
    score: float
    search_mode: str
    char_len: int | None = None
    n_chunks: int | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    results: list[SearchResult]
    latency_ms: float
    mode: str


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        repository: DataRepository | None = None,
        store: QdrantStore | None = None,
        bm25: BM25Index | None = None,
        encoder_v1: TextEncoder | None = None,
        encoder_v2: TextEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or DataRepository(settings.data_path, settings.corpus_path)
        self._store = store
        self._bm25 = bm25
        self._encoder_v1 = encoder_v1
        self._encoder_v2 = encoder_v2

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.settings.bm25_dir)
            expected = self._bm25.metadata.get("fingerprint")
            if expected and expected != self.repository.fingerprint():
                raise RuntimeError("BM25 index was built for a different mapping file")
        return self._bm25

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore(self.settings.qdrant_url, self.settings.qdrant_api_key)
        return self._store

    def _encoder(self, iteration: int) -> TextEncoder:
        if iteration == 1:
            if self._encoder_v1 is None:
                self._encoder_v1 = create_encoder(
                    self.settings.model_v1,
                    self.settings.device,
                    self.settings.trust_remote_code,
                )
            return self._encoder_v1
        if self._encoder_v2 is None:
            self._encoder_v2 = create_encoder(
                self.settings.model_v2,
                self.settings.device,
                self.settings.trust_remote_code,
            )
        return self._encoder_v2

    def _dense_ranked(
        self, query: str, iteration: int, limit: int, sources: set[str] | None
    ) -> list[RankedItem]:
        encoder = self._encoder(iteration)
        vector = encoder.encode([query])[0]
        collection = (
            self.settings.qdrant_collection_v1
            if iteration == 1
            else self.settings.qdrant_collection_v2
        )
        expected_dimension = self.store.collection_dimension(collection)
        if vector.shape[0] != expected_dimension:
            raise RuntimeError(
                f"Query model produces {vector.shape[0]} dimensions, but {collection} expects "
                f"{expected_dimension}. Use the same model that produced the document embeddings."
            )
        hits = self.store.search(
            collection,
            vector,
            limit=limit,
            source_filter=sources,
            exact=iteration == 1,
            hnsw_ef=self.settings.hnsw_ef_search,
        )
        return [RankedItem(hit.doc_id, hit.row_index, hit.score) for hit in hits]

    def _bm25_ranked(
        self, query: str, limit: int, sources: set[str] | None
    ) -> list[RankedItem]:
        hits = self.bm25.search(query, limit=limit, source_filter=sources)
        return [
            RankedItem(self.repository.document(hit.row_index).doc_id, hit.row_index, hit.score)
            for hit in hits
        ]

    @staticmethod
    def _deduplicate(items: list[RankedItem], limit: int) -> list[RankedItem]:
        seen: set[str] = set()
        result: list[RankedItem] = []
        for item in items:
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            result.append(item)
            if len(result) == limit:
                break
        return result

    def search(
        self,
        query: str,
        *,
        mode: SearchMode | str | None = None,
        top_k: int = 10,
        sources: set[str] | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if top_k < 1 or top_k > self.settings.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.settings.max_top_k}")
        if sources:
            unknown = sources.difference(self.repository.sources)
            if unknown:
                raise ValueError(f"Unknown sources: {sorted(unknown)}")

        selected = SearchMode(mode or self.settings.default_mode)
        candidates = max(top_k * 4, self.settings.search_candidates)
        started = time.perf_counter()
        if selected == SearchMode.DENSE_V1:
            ranked = self._deduplicate(self._dense_ranked(query, 1, candidates, sources), top_k)
        elif selected == SearchMode.DENSE_V2:
            ranked = self._deduplicate(self._dense_ranked(query, 2, candidates, sources), top_k)
        elif selected == SearchMode.BM25:
            ranked = self._deduplicate(self._bm25_ranked(query, candidates, sources), top_k)
        else:
            iteration = 1 if selected == SearchMode.HYBRID_V1 else 2
            ranked = reciprocal_rank_fusion(
                [
                    self._dense_ranked(query, iteration, candidates, sources),
                    self._bm25_ranked(query, candidates, sources),
                ],
                rrf_k=self.settings.rrf_k,
                limit=top_k,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        results: list[SearchResult] = []
        for rank, item in enumerate(ranked, start=1):
            doc = self.repository.document(item.row_index)
            snippet = None
            if doc.text:
                compact = " ".join(doc.text.split())
                snippet = compact[:400] + ("…" if len(compact) > 400 else "")
            results.append(
                SearchResult(
                    rank=rank,
                    row_index=doc.row_index,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    score=float(item.score),
                    search_mode=selected.value,
                    char_len=doc.char_len,
                    n_chunks=doc.n_chunks,
                    snippet=snippet,
                )
            )
        return SearchResponse(results=results, latency_ms=latency_ms, mode=selected.value)

    def health(self) -> dict:
        collections = {}
        for key, name in (
            ("v1", self.settings.qdrant_collection_v1),
            ("v2", self.settings.qdrant_collection_v2),
        ):
            try:
                collections[key] = self.store.collection_stats(name)
            except Exception as exc:
                collections[key] = {"status": "unavailable", "detail": str(exc)}
        bm25_status = "ready"
        try:
            _ = self.bm25
        except Exception as exc:
            bm25_status = f"unavailable: {exc}"
        return {
            "status": "ok" if any(v.get("status") != "unavailable" for v in collections.values()) else "degraded",
            "documents": len(self.repository),
            "sources": self.repository.sources,
            "bm25": bm25_status,
            "collections": collections,
        }
