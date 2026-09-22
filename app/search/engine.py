from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .bm25 import BM25Index
from .config import Settings
from .data import DataRepository
from .encoders import TextEncoder, create_encoder
from .fusion import RankedItem, reciprocal_rank_fusion
from .precomputed import (
    ChunkHit,
    PrecomputedBM25SIndex,
    PrecomputedDenseIndex,
    PrecomputedSPLADEIndex,
)
from .qdrant_store import QdrantStore
from .rerankers import CrossEncoderReranker
from .splade import SPLADEIndex, create_sparse_encoder


class SearchMode(StrEnum):
    DENSE_V1 = "dense_v1"
    DENSE_V2 = "dense_v2"
    BM25 = "bm25"
    SPLADE = "splade"
    HYBRID_V1 = "hybrid_v1"
    HYBRID_V2 = "hybrid_v2"
    TRIPLE_HYBRID_V1 = "triple_hybrid_v1"
    TRIPLE_HYBRID_V2 = "triple_hybrid_v2"


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    row_index: int
    doc_id: str
    chunk_id: str | None
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
        splade_index: SPLADEIndex | None = None,
        splade_encoder=None,
        precomputed_bm25=None,
        precomputed_dense=None,
        precomputed_splade=None,
        reranker: CrossEncoderReranker | None = None,
        encoder_v1: TextEncoder | None = None,
        encoder_v2: TextEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or DataRepository(
            settings.data_path,
            settings.corpus_path,
            settings.chunks_path,
        )
        self._store = store
        self._bm25 = bm25
        self._splade_index = splade_index
        self._splade_encoder = splade_encoder
        self._precomputed_bm25 = precomputed_bm25
        self._precomputed_dense = precomputed_dense
        self._precomputed_splade = precomputed_splade
        self._reranker = reranker
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
    def precomputed_bm25(self) -> PrecomputedBM25SIndex:
        if self._precomputed_bm25 is None:
            self._precomputed_bm25 = PrecomputedBM25SIndex(
                self.settings.precomputed_bm25s_dir,
                stopwords=self.settings.bm25s_stopwords,
                stemmer_name=self.settings.bm25s_stemmer,
            )
        return self._precomputed_bm25

    @property
    def precomputed_dense(self) -> PrecomputedDenseIndex:
        if self._precomputed_dense is None:
            self._precomputed_dense = PrecomputedDenseIndex(self.settings.precomputed_dense_dir)
        return self._precomputed_dense

    @property
    def precomputed_splade(self) -> PrecomputedSPLADEIndex:
        if self._precomputed_splade is None:
            self._precomputed_splade = PrecomputedSPLADEIndex(self.settings.precomputed_splade_dir)
        return self._precomputed_splade

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore(self.settings.qdrant_url, self.settings.qdrant_api_key)
        return self._store

    @property
    def splade_index(self) -> SPLADEIndex:
        if self._splade_index is None:
            self._splade_index = SPLADEIndex.load(self.settings.splade_dir)
            expected = self._splade_index.metadata.get("fingerprint")
            if expected and expected != self.repository.fingerprint():
                raise RuntimeError("SPLADE index was built for a different mapping file")
        return self._splade_index

    @property
    def splade_encoder(self):
        if self._splade_encoder is None:
            self._splade_encoder = create_sparse_encoder(
                self.settings.splade_model,
                device=self.settings.device,
                max_length=self.settings.splade_max_length,
                threshold=self.settings.splade_threshold,
                batch_size=self.settings.splade_batch_size,
                fp16=self.settings.splade_fp16,
            )
        return self._splade_encoder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if not self.settings.reranker_enabled and self._reranker is None:
            raise ValueError("Reranker is disabled. Set RERANKER_ENABLED=true to enable it.")
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.settings.reranker_model,
                device=self.settings.device,
                batch_size=self.settings.reranker_batch_size,
                max_length=self.settings.reranker_max_length,
                trust_remote_code=self.settings.reranker_trust_remote_code,
            )
        return self._reranker

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
        encoded_query = f"{self.settings.dense_query_prefix}{query}"
        vector = encoder.encode([encoded_query])[0]
        if self.settings.dense_normalize_query:
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
        if self.settings.use_precomputed_indexes:
            hits = self.precomputed_dense.search(vector, limit=limit)
            return self._chunk_hits_to_ranked(hits, limit, sources)
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
        if self.settings.use_precomputed_indexes:
            hits = self.precomputed_bm25.search(query, limit=self._external_limit(limit))
            return self._chunk_hits_to_ranked(hits, limit, sources)
        hits = self.bm25.search(query, limit=limit, source_filter=sources)
        return [
            RankedItem(self.repository.document(hit.row_index).doc_id, hit.row_index, hit.score)
            for hit in hits
        ]

    def _splade_ranked(
        self, query: str, limit: int, sources: set[str] | None
    ) -> list[RankedItem]:
        query_matrix, order = self.splade_encoder.encode(
            [query],
            batch_size=self.settings.splade_query_batch_size,
            sort_by_length=False,
            show_progress=False,
        )
        if not np.array_equal(order, np.arange(1)):
            raise RuntimeError("SPLADE query ordering must stay unchanged")
        if self.settings.use_precomputed_indexes:
            hits = self.precomputed_splade.search(
                query_matrix,
                limit=self._external_limit(limit),
            )
            return self._chunk_hits_to_ranked(hits, limit, sources)
        hits = self.splade_index.search(query_matrix, limit=limit, source_filter=sources)
        return [
            RankedItem(self.repository.document(hit.row_index).doc_id, hit.row_index, hit.score)
            for hit in hits
        ]

    @staticmethod
    def _external_limit(limit: int) -> int:
        return max(limit * 4, limit)

    def _chunk_hits_to_ranked(
        self,
        hits: list[ChunkHit],
        limit: int,
        sources: set[str] | None,
    ) -> list[RankedItem]:
        ranked: list[RankedItem] = []
        seen_chunks: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(hit.chunk_id)
            doc = self.repository.document_for_chunk(hit.chunk_id)
            if doc is None:
                continue
            if sources and doc.source not in sources:
                continue
            ranked.append(
                RankedItem(
                    doc_id=doc.doc_id,
                    row_index=doc.row_index,
                    score=hit.score,
                    chunk_id=hit.chunk_id,
                )
            )
            if len(ranked) == limit:
                break
        return ranked

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
        rerank: bool = False,
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
        if rerank:
            candidates = max(candidates, self.settings.reranker_candidates)
        started = time.perf_counter()
        if selected == SearchMode.DENSE_V1:
            ranked = self._deduplicate(self._dense_ranked(query, 1, candidates, sources), candidates)
        elif selected == SearchMode.DENSE_V2:
            ranked = self._deduplicate(self._dense_ranked(query, 2, candidates, sources), candidates)
        elif selected == SearchMode.BM25:
            ranked = self._deduplicate(self._bm25_ranked(query, candidates, sources), candidates)
        elif selected == SearchMode.SPLADE:
            ranked = self._deduplicate(self._splade_ranked(query, candidates, sources), candidates)
        else:
            if selected in (SearchMode.HYBRID_V1, SearchMode.TRIPLE_HYBRID_V1):
                iteration = 1
            else:
                iteration = 2
            rankings = [
                self._dense_ranked(query, iteration, candidates, sources),
                self._bm25_ranked(query, candidates, sources),
            ]
            if selected in (SearchMode.TRIPLE_HYBRID_V1, SearchMode.TRIPLE_HYBRID_V2):
                rankings.append(self._splade_ranked(query, candidates, sources))
            ranked = reciprocal_rank_fusion(
                rankings,
                rrf_k=self.settings.rrf_k,
                limit=candidates,
            )
        if rerank:
            ranked = self.reranker.rerank(query, ranked, self._rerank_text, top_k=top_k)
        else:
            ranked = ranked[:top_k]
        latency_ms = (time.perf_counter() - started) * 1000
        response_mode = f"{selected.value}+rerank" if rerank else selected.value

        results: list[SearchResult] = []
        for rank, item in enumerate(ranked, start=1):
            doc = self.repository.document(item.row_index)
            result_text = self.repository.chunk_text(item.chunk_id) if item.chunk_id else doc.text
            snippet = None
            if result_text:
                compact = " ".join(result_text.split())
                snippet = compact[:400] + ("…" if len(compact) > 400 else "")
            results.append(
                SearchResult(
                    rank=rank,
                    row_index=doc.row_index,
                    doc_id=doc.doc_id,
                    chunk_id=item.chunk_id,
                    title=doc.title,
                    source=doc.source,
                    score=float(item.score),
                    search_mode=response_mode,
                    char_len=doc.char_len,
                    n_chunks=doc.n_chunks,
                    snippet=snippet,
                )
            )
        return SearchResponse(results=results, latency_ms=latency_ms, mode=response_mode)

    def _rerank_text(self, row_index: int) -> str:
        doc = self.repository.document(row_index)
        if doc.text:
            return f"{doc.title}\n{doc.text}".strip()
        return doc.title

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
            _ = self.precomputed_bm25 if self.settings.use_precomputed_indexes else self.bm25
        except Exception as exc:
            bm25_status = f"unavailable: {exc}"
        splade_status = "ready"
        try:
            _ = self.precomputed_splade if self.settings.use_precomputed_indexes else self.splade_index
        except Exception as exc:
            splade_status = f"unavailable: {exc}"
        dense_status = "ready"
        try:
            if self.settings.use_precomputed_indexes:
                _ = self.precomputed_dense
        except Exception as exc:
            dense_status = f"unavailable: {exc}"
        external_ready = (
            self.settings.use_precomputed_indexes
            and bm25_status == "ready"
            and dense_status == "ready"
            and splade_status == "ready"
        )
        qdrant_ready = any(v.get("status") != "unavailable" for v in collections.values())
        return {
            "status": "ok" if external_ready or qdrant_ready else "degraded",
            "documents": len(self.repository),
            "sources": self.repository.sources,
            "bm25": bm25_status,
            "dense": dense_status,
            "splade": splade_status,
            "reranker_enabled": self.settings.reranker_enabled,
            "use_precomputed_indexes": self.settings.use_precomputed_indexes,
            "collections": collections,
        }
