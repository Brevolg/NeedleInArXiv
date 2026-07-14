from pathlib import Path

import pandas as pd

from search.bm25 import BM25Index
from search.config import Settings
from search.data import DataRepository
from search.encoders import HashingEncoder
from search.engine import SearchEngine
from search.qdrant_store import DenseHit


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

