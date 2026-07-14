import pytest

from search.metrics import average_precision, ndcg_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking_metrics_are_one():
    ranking = ["a", "b", "x"]
    relevant = {"a", "b"}
    assert recall_at_k(ranking, relevant, 2) == 1.0
    assert average_precision(ranking, relevant, 2) == 1.0
    assert ndcg_at_k(ranking, relevant, 2) == 1.0
    assert reciprocal_rank(ranking, relevant, 10) == 1.0


def test_duplicate_retrieval_does_not_count_twice():
    assert recall_at_k(["a", "a", "x"], {"a", "b"}, 3) == 0.5


def test_empty_qrels_are_not_silently_scored():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 10)

