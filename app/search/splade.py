from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .bm25 import tokenize


@dataclass(frozen=True, slots=True)
class SPLADEHit:
    row_index: int
    score: float


class HashingSparseEncoder:
    """Small deterministic sparse encoder for tests and demo assets only."""

    def __init__(self, dimensions: int = 2048) -> None:
        self.dimensions = dimensions
        self.vectorizer = HashingVectorizer(
            analyzer=tokenize,
            n_features=dimensions,
            alternate_sign=False,
            norm=None,
            dtype=np.float32,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        sort_by_length: bool = True,
        show_progress: bool = False,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        del batch_size, show_progress
        order = _length_order(texts) if sort_by_length else np.arange(len(texts))
        sorted_texts = [texts[i] for i in order]
        matrix = self.vectorizer.transform(sorted_texts).tocsr()
        matrix.sum_duplicates()
        return matrix, order


class SPLADETextEncoder:
    """SPLADE encoder following the ordering contract from notebooks/eval_splade.py."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        *,
        device: str = "cpu",
        max_length: int = 374,
        threshold: float = 0.01,
        batch_size: int = 32,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.max_length = max_length
        self.threshold = threshold
        self.batch_size = batch_size
        self.fp16 = fp16
        self._tokenizer = None
        self._model = None
        self._torch = None

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def torch(self):
        self._ensure_loaded()
        return self._torch

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self.device_name

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = self.device_name
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device_name = device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        if device == "cuda" and self.fp16:
            model = model.half()
        model = model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        sort_by_length: bool = True,
        show_progress: bool = False,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        if not texts:
            return sparse.csr_matrix((0, self.tokenizer.vocab_size), dtype=np.float32), np.array([])

        order = _length_order(texts) if sort_by_length else np.arange(len(texts))
        sorted_texts = [str(texts[i]) for i in order]
        batch_size = batch_size or self.batch_size

        iterator = range(0, len(sorted_texts), batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(
                iterator,
                total=(len(sorted_texts) + batch_size - 1) // batch_size,
                desc="SPLADE encoding",
            )

        all_counts: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        torch = self.torch
        with torch.inference_mode():
            for start in iterator:
                batch = sorted_texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                logits = self.model(**inputs).logits
                max_logits = torch.max(logits, dim=1).values
                weights = torch.log1p(torch.relu(max_logits))

                mask = weights > self.threshold
                rows, cols = torch.where(mask)
                vals = weights[mask]

                if vals.numel():
                    counts = torch.bincount(rows, minlength=len(batch))
                    all_counts.append(counts.cpu().numpy().astype(np.int64))
                    all_indices.append(cols.cpu().numpy().astype(np.int32))
                    all_values.append(vals.float().cpu().numpy().astype(np.float32))
                else:
                    all_counts.append(np.zeros(len(batch), dtype=np.int64))

                del inputs, logits, max_logits, weights, mask, rows, cols, vals
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        counts = np.concatenate(all_counts)
        indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
        values = np.concatenate(all_values) if all_values else np.array([], dtype=np.float32)
        matrix = sparse.csr_matrix(
            (values, indices, indptr),
            shape=(len(sorted_texts), self.tokenizer.vocab_size),
            dtype=np.float32,
        )
        return matrix, order


class SPLADEIndex:
    """Persisted sparse index where matrix rows are aligned with row_indices."""

    def __init__(
        self,
        matrix: sparse.csr_matrix,
        row_indices: np.ndarray,
        sources: np.ndarray,
        metadata: dict[str, str | int | float | bool],
    ) -> None:
        self.matrix = matrix.tocsr().astype(np.float32)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.sources = np.asarray(sources).astype(str)
        self.metadata = metadata
        if self.matrix.shape[0] != len(self.row_indices):
            raise ValueError("SPLADE matrix/row_indices row mismatch")
        if self.matrix.shape[0] != len(self.sources):
            raise ValueError("SPLADE matrix/source row mismatch")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "matrix.npz", self.matrix, compressed=True)
        np.save(directory / "row_indices.npy", self.row_indices, allow_pickle=False)
        np.save(directory / "sources.npy", self.sources, allow_pickle=False)
        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "SPLADEIndex":
        matrix_path = directory / "matrix.npz"
        row_indices_path = directory / "row_indices.npy"
        if not matrix_path.exists() and (directory / "splade_index.npz").exists():
            matrix_path = directory / "splade_index.npz"
        if not row_indices_path.exists() and (directory / "splade_chunk_ids.npy").exists():
            row_indices_path = directory / "splade_chunk_ids.npy"

        required = [matrix_path, row_indices_path, directory / "sources.npy", directory / "metadata.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"SPLADE index is incomplete, missing: {missing}")

        return cls(
            matrix=sparse.load_npz(matrix_path).tocsr(),
            row_indices=np.load(row_indices_path, allow_pickle=False),
            sources=np.load(directory / "sources.npy", allow_pickle=False),
            metadata=json.loads((directory / "metadata.json").read_text("utf-8")),
        )

    def search(
        self,
        query_vector: sparse.csr_matrix,
        *,
        limit: int = 10,
        source_filter: set[str] | None = None,
    ) -> list[SPLADEHit]:
        if limit < 1:
            return []
        query_vector = query_vector.tocsr()
        if query_vector.shape[0] != 1:
            raise ValueError("SPLADE search expects one query vector")
        if query_vector.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                f"SPLADE query has {query_vector.shape[1]} features, index expects "
                f"{self.matrix.shape[1]}"
            )

        scores = (query_vector @ self.matrix.T).toarray().ravel()
        valid = scores > 0
        if source_filter:
            valid &= np.isin(self.sources, list(source_filter))
        valid_rows = np.flatnonzero(valid)
        if not len(valid_rows):
            return []

        take = min(limit, len(valid_rows))
        if take < len(valid_rows):
            local = np.argpartition(scores[valid_rows], -take)[-take:]
            chosen = valid_rows[local]
        else:
            chosen = valid_rows

        chosen = chosen[np.lexsort((self.row_indices[chosen], -scores[chosen]))]
        return [
            SPLADEHit(row_index=int(self.row_indices[matrix_row]), score=float(scores[matrix_row]))
            for matrix_row in chosen[:take]
        ]


def create_sparse_encoder(
    model_name: str,
    *,
    device: str = "cpu",
    max_length: int = 374,
    threshold: float = 0.01,
    batch_size: int = 32,
    fp16: bool = True,
):
    if model_name.startswith("hashing_sparse:"):
        return HashingSparseEncoder(int(model_name.split(":", 1)[1]))
    return SPLADETextEncoder(
        model_name,
        device=device,
        max_length=max_length,
        threshold=threshold,
        batch_size=batch_size,
        fp16=fp16,
    )


def _length_order(texts: Sequence[str]) -> np.ndarray:
    return np.argsort([len(str(text).split()) for text in texts])
