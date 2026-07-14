from pathlib import Path

import pandas as pd

from search.bm25 import BM25Index
from search.config import Settings
from search.data import DataRepository
from search.encoders import HashingEncoder
from search.engine import SearchEngine
from search.qdrant_store import DenseHit
from search.splade import SPLADEHit


class FakeStore:
    def collection_dimension(self, name: str) -> int:
        return 32

    def search(self, name, vector, **kwargs):
        return [
            DenseHit(0, 0, "d1", 0.9, {"doc_id": "d1", "row_index": 0}),
            DenseHit(1, 1, "d2", 0.8, {"doc_id": "d2", "row_index": 1}),
        ]

    def collection_stats(self, name: str):
        return {"status": "green", "points_count": 2, "dimension": 32}


class FakeSparseEncoder:
    def encode(self, texts, **kwargs):
        import numpy as np
        from scipy import sparse

        assert kwargs["sort_by_length"] is False
        return sparse.csr_matrix([[1.0, 0.0]], dtype=np.float32), np.arange(len(texts))


class FakeSPLADEIndex:
    metadata = {"fingerprint": ""}

    def search(self, query_matrix, *, limit, source_filter=None):
        return [
            SPLADEHit(1, 2.0),
            SPLADEHit(0, 1.0),
        ][:limit]


def test_hybrid_engine_returns_ranked_metadata(tmp_path: Path):
    mapping_path = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "source": ["github", "jira"],
            "title": ["multipart upload limits", "unrelated incident"],
            "char_len": [100, 200],
            "n_chunks": [1, 2],
        }
    ).to_parquet(mapping_path, index=False)
    repository = DataRepository(mapping_path)
    bm25 = BM25Index.build(
        repository.search_texts(), repository.frame["source"].tolist(), fingerprint=repository.fingerprint()
    )
    engine = SearchEngine(
        Settings(data_path=mapping_path, model_v1="hashing:32"),
        repository=repository,
        store=FakeStore(),
        bm25=bm25,
        encoder_v1=HashingEncoder(32),
    )
    response = engine.search("upload limits", mode="hybrid_v1", top_k=2)
    assert response.results[0].doc_id == "d1"
    assert response.results[0].rank == 1
    assert response.latency_ms >= 0


def test_triple_hybrid_uses_splade_ranker(tmp_path: Path):
    mapping_path = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "source": ["github", "jira"],
            "title": ["multipart upload limits", "unrelated incident"],
        }
    ).to_parquet(mapping_path, index=False)
    repository = DataRepository(mapping_path)
    bm25 = BM25Index.build(
        repository.search_texts(),
        repository.frame["source"].tolist(),
        fingerprint=repository.fingerprint(),
    )
    engine = SearchEngine(
        Settings(data_path=mapping_path, model_v1="hashing:32"),
        repository=repository,
        store=FakeStore(),
        bm25=bm25,
        splade_index=FakeSPLADEIndex(),
        splade_encoder=FakeSparseEncoder(),
        encoder_v1=HashingEncoder(32),
    )

    response = engine.search("upload limits", mode="triple_hybrid_v1", top_k=2)

    assert response.mode == "triple_hybrid_v1"
    assert [result.doc_id for result in response.results] == ["d1", "d2"]
