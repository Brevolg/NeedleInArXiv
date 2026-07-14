from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


def _unique_at_k(ranked_ids: list[str], k: int) -> list[str]:
    return list(dict.fromkeys(ranked_ids))[:k]


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        raise ValueError("Recall is undefined for a query without relevant documents")
    return len(set(_unique_at_k(ranked_ids, k)) & relevant_ids) / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str], k: int | None = None) -> float:
    if not relevant_ids:
        raise ValueError("Average precision is undefined without relevant documents")
    ranking = list(dict.fromkeys(ranked_ids))
    if k is not None:
        ranking = ranking[:k]
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / rank
    denominator = min(len(relevant_ids), k) if k is not None else len(relevant_ids)
    return total / denominator if denominator else 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        raise ValueError("nDCG is undefined for a query without relevant documents")
    ranking = _unique_at_k(ranked_ids, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranking, start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    if not relevant_ids:
        raise ValueError("Reciprocal rank is undefined without relevant documents")
    for rank, doc_id in enumerate(_unique_at_k(ranked_ids, k), start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


@dataclass(slots=True)
class EvaluationAccumulator:
    ndcg: list[float]
    recall: list[float]
    ap: list[float]
    rr: list[float]
    latencies_ms: list[float]
    no_answer_queries: int = 0
    no_answer_empty_results: int = 0

    @classmethod
    def create(cls) -> "EvaluationAccumulator":
        return cls([], [], [], [], [])

    def add(
        self,
        ranked_ids: list[str],
        relevant_ids: set[str],
        latency_ms: float,
        k: int = 10,
    ) -> None:
        self.latencies_ms.append(latency_ms)
        if not relevant_ids:
            self.no_answer_queries += 1
            self.no_answer_empty_results += int(not ranked_ids)
            return
        self.ndcg.append(ndcg_at_k(ranked_ids, relevant_ids, k))
        self.recall.append(recall_at_k(ranked_ids, relevant_ids, k))
        self.ap.append(average_precision(ranked_ids, relevant_ids, k))
        self.rr.append(reciprocal_rank(ranked_ids, relevant_ids, k))

    def summary(self) -> dict[str, float | int | None]:
        return {
            "evaluated_queries": len(self.ndcg),
            "no_answer_queries": self.no_answer_queries,
            "ndcg_at_10": fmean(self.ndcg) if self.ndcg else None,
            "recall_at_10": fmean(self.recall) if self.recall else None,
            "map_at_10": fmean(self.ap) if self.ap else None,
            "mrr_at_10": fmean(self.rr) if self.rr else None,
            "latency_mean_ms": fmean(self.latencies_ms) if self.latencies_ms else None,
            "no_answer_empty_rate": (
                self.no_answer_empty_results / self.no_answer_queries
                if self.no_answer_queries
                else None
            ),
        }
