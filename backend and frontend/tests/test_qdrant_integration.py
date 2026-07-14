from pathlib import Path

import numpy as np
import pandas as pd

from search.data import DataRepository
from search.qdrant_store import QdrantStore


def test_in_memory_qdrant_round_trip(tmp_path: Path):
    mapping_path = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "doc_id": ["d0", "d1", "d2"],
            "source": ["github", "jira", "github"],
            "title": ["alpha", "beta", "gamma"],
            "char_len": [10, 20, 30],
            "n_chunks": [1, 1, 1],
        }
    ).to_parquet(mapping_path, index=False)
    repository = DataRepository(mapping_path)
    vectors = np.eye(3, dtype=np.float32)
    store = QdrantStore(":memory:")
    store.create_collection("test", 3, recreate=False, hnsw=False)
    store.upload("test", vectors, repository, batch_size=2)
    hits = store.search("test", vectors[1], limit=2, exact=True)
    assert hits[0].doc_id == "d1"
    filtered = store.search(
        "test", vectors[0], limit=3, source_filter={"jira"}, exact=True
    )
    assert [hit.doc_id for hit in filtered] == ["d1"]
    assert store.collection_stats("test")["points_count"] == 3

