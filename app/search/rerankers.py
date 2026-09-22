from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .fusion import RankedItem


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 4,
        max_length: int = 256,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RankedItem],
        text_for_row: Callable[[int], str],
        *,
        top_k: int,
    ) -> list[RankedItem]:
        pairs: list[tuple[str, str]] = []
        valid: list[RankedItem] = []
        skipped: list[RankedItem] = []
        for item in candidates:
            text = text_for_row(item.row_index).strip()
            if not text:
                skipped.append(item)
                continue
            pairs.append((query, text))
            valid.append(item)

        if not pairs:
            return list(candidates[:top_k])

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores).reshape(-1)
        order = np.argsort(scores)[::-1]
        reranked = [
            RankedItem(
                doc_id=valid[index].doc_id,
                row_index=valid[index].row_index,
                score=float(scores[index]),
            )
            for index in order[:top_k]
        ]
        if len(reranked) < top_k:
            reranked.extend(skipped[: top_k - len(reranked)])
        return reranked
