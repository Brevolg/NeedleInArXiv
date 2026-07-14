from search.fusion import RankedItem, reciprocal_rank_fusion


def item(doc_id: str, row: int, score: float = 1.0) -> RankedItem:
    return RankedItem(doc_id=doc_id, row_index=row, score=score)


def test_rrf_rewards_documents_present_in_both_rankings():
    dense = [item("a", 0), item("b", 1), item("c", 2)]
    lexical = [item("b", 1), item("d", 3), item("a", 0)]
    result = reciprocal_rank_fusion([dense, lexical], rrf_k=60, limit=4)
    assert result[0].doc_id == "b"
    assert {entry.doc_id for entry in result} == {"a", "b", "c", "d"}


def test_rrf_ignores_duplicate_doc_ids_inside_one_ranking():
    result = reciprocal_rank_fusion(
        [[item("a", 0), item("a", 1)], [item("b", 2)]], rrf_k=1, limit=5
    )
    assert [entry.doc_id for entry in result].count("a") == 1

