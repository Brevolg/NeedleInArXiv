from pathlib import Path

from search.bm25 import BM25Index


def test_bm25_build_search_save_load(tmp_path: Path):
    texts = [
        "semantic search with dense vectors",
        "database backup and disaster recovery",
        "dense retrieval fused with keyword search",
    ]
    index = BM25Index.build(texts, ["a", "b", "a"], fingerprint="test")
    hits = index.search("dense search", limit=3)
    assert hits[0].row_index in {0, 2}
    assert all(hit.score > 0 for hit in hits)

    directory = tmp_path / "bm25"
    index.save(directory)
    restored = BM25Index.load(directory)
    assert restored.search("disaster recovery", limit=1)[0].row_index == 1
    assert restored.search("dense", limit=5, source_filter={"b"}) == []

