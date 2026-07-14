from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankedItem:
    doc_id: str
    row_index: int
    score: float


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[RankedItem]], *, rrf_k: int = 60, limit: int = 10
) -> list[RankedItem]:
    """Fuse rankings by document id and retain the best metadata row for each id."""
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, RankedItem] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            scores[item.doc_id] += 1.0 / (rrf_k + rank)
            current = representatives.get(item.doc_id)
            if current is None or item.score > current.score:
                representatives[item.doc_id] = item

    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
    return [
        RankedItem(
            doc_id=doc_id,
            row_index=representatives[doc_id].row_index,
            score=scores[doc_id],
        )
        for doc_id in ordered
    ]

