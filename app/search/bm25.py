from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer shared by BM25 and the hashing sparse encoder."""
    return _TOKEN_RE.findall(str(text).lower())


@dataclass(frozen=True, slots=True)
class BM25Hit:
    row_index: int
    score: float


class BM25Index:
    """Okapi BM25 stored as a CSR matrix of saturated term frequencies.

    matrix[d, t] = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_doc_len));
    a query is scored as the idf-weighted sum of its terms' columns.
    """

    def __init__(
        self,
        matrix: sparse.csr_matrix,
        idf: np.ndarray,
        vocabulary: dict[str, int],
        sources: np.ndarray,
        metadata: dict[str, str | int | float],
    ) -> None:
        self.matrix = matrix.tocsr().astype(np.float32)
        self.idf = np.asarray(idf, dtype=np.float32)
        self.vocabulary = vocabulary
        self.sources = np.asarray(sources).astype(str)
        self.metadata = metadata
        if self.matrix.shape[1] != len(self.idf) or len(self.idf) != len(self.vocabulary):
            raise ValueError("BM25 matrix/idf/vocabulary feature mismatch")
        if self.matrix.shape[0] != len(self.sources):
            raise ValueError("BM25 matrix/source row mismatch")

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        sources: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        min_df: int = 1,
        fingerprint: str = "",
        content_kind: str = "title",
    ) -> BM25Index:
        if len(texts) != len(sources):
            raise ValueError("texts and sources must have the same length")
        term_counts = [Counter(tokenize(text)) for text in texts]
        doc_freq: Counter[str] = Counter()
        for counts in term_counts:
            doc_freq.update(counts.keys())
        terms = sorted(term for term, df in doc_freq.items() if df >= min_df)
        vocabulary = {term: column for column, term in enumerate(terms)}

        doc_lengths = np.array([sum(counts.values()) for counts in term_counts], dtype=np.float64)
        avg_doc_len = float(doc_lengths.mean()) if len(doc_lengths) else 0.0
        norm = 1 - b + b * doc_lengths / avg_doc_len if avg_doc_len else np.ones_like(doc_lengths)

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for row, counts in enumerate(term_counts):
            for term, tf in counts.items():
                column = vocabulary.get(term)
                if column is None:
                    continue
                rows.append(row)
                cols.append(column)
                values.append(tf * (k1 + 1) / (tf + k1 * norm[row]))
        matrix = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(len(texts), len(vocabulary)),
            dtype=np.float32,
        )

        n_docs = len(texts)
        df = np.array([doc_freq[term] for term in terms], dtype=np.float64)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        metadata: dict[str, str | int | float] = {
            "version": 1,
            "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "k1": k1,
            "b": b,
            "avg_doc_len": avg_doc_len,
            "fingerprint": fingerprint,
            "content_kind": content_kind,
        }
        return cls(matrix, idf, vocabulary, np.asarray(sources), metadata)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "matrix.npz", self.matrix, compressed=True)
        np.save(directory / "idf.npy", self.idf, allow_pickle=False)
        np.save(directory / "sources.npy", self.sources, allow_pickle=False)
        (directory / "vocabulary.json").write_text(
            json.dumps(self.vocabulary, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        required = ["matrix.npz", "idf.npy", "sources.npy", "vocabulary.json", "metadata.json"]
        missing = [str(directory / name) for name in required if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"BM25 index is incomplete, missing: {missing}")
        return cls(
            matrix=sparse.load_npz(directory / "matrix.npz").tocsr(),
            idf=np.load(directory / "idf.npy", allow_pickle=False),
            vocabulary=json.loads((directory / "vocabulary.json").read_text("utf-8")),
            sources=np.load(directory / "sources.npy", allow_pickle=False),
            metadata=json.loads((directory / "metadata.json").read_text("utf-8")),
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source_filter: set[str] | None = None,
    ) -> list[BM25Hit]:
        if limit < 1:
            return []
        columns = sorted({self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary})
        if not columns:
            return []

        scores = np.asarray(self.matrix[:, columns] @ self.idf[columns]).ravel()
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

        chosen = chosen[np.lexsort((chosen, -scores[chosen]))]
        return [BM25Hit(row_index=int(row), score=float(scores[row])) for row in chosen[:take]]
