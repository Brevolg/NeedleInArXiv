from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_MAPPING_COLUMNS = {"doc_id", "source", "title"}


@dataclass(frozen=True, slots=True)
class Document:
    row_index: int
    doc_id: str
    source: str
    title: str
    chunk_id: str | None = None
    char_len: int | None = None
    n_chunks: int | None = None
    text: str | None = None


class DataRepository:
    """Keeps row order intact because embedding row i belongs to mapping row i."""

    def __init__(
        self,
        mapping_path: Path,
        corpus_path: Path | None = None,
        chunks_path: Path | None = None,
    ) -> None:
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        frame = pd.read_parquet(mapping_path)
        missing = REQUIRED_MAPPING_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Mapping is missing required columns: {sorted(missing)}")
        if frame[list(REQUIRED_MAPPING_COLUMNS)].isna().any().any():
            raise ValueError("Mapping contains nulls in doc_id/source/title")

        self.mapping_path = mapping_path
        self.frame = frame.reset_index(drop=True)
        self.frame["doc_id"] = self.frame["doc_id"].astype(str)
        self.frame["source"] = self.frame["source"].astype(str)
        self.frame["title"] = self.frame["title"].astype(str)
        self._content_by_id: dict[str, str] = {}
        self._doc_row_by_id: dict[str, int] = {}
        for row_index, doc_id in enumerate(self.frame["doc_id"].astype(str)):
            self._doc_row_by_id.setdefault(str(doc_id), int(row_index))
        self._chunk_to_doc_id: dict[str, str] = {}
        self._chunk_to_text: dict[str, str] = {}
        self.content_column: str | None = None
        if corpus_path is not None:
            self._load_corpus(corpus_path)
        if chunks_path is not None:
            self._load_chunks(chunks_path)
        elif "chunk_id" in self.frame:
            self._load_chunks_from_frame(self.frame)

    def _load_corpus(self, corpus_path: Path) -> None:
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        corpus = pd.read_parquet(corpus_path)
        if "doc_id" not in corpus:
            raise ValueError("Corpus must contain doc_id")
        candidates = [c for c in ("text", "abstract", "content", "body") if c in corpus]
        if not candidates:
            raise ValueError("Corpus must contain one of: text, abstract, content, body")
        self.content_column = candidates[0]
        clean = corpus[["doc_id", self.content_column]].dropna().drop_duplicates("doc_id")
        self._content_by_id = dict(
            zip(clean["doc_id"].astype(str), clean[self.content_column].astype(str), strict=False)
        )

    def _load_chunks(self, chunks_path: Path) -> None:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        chunks = pd.read_parquet(chunks_path)
        self._load_chunks_from_frame(chunks)

    def _load_chunks_from_frame(self, chunks: pd.DataFrame) -> None:
        if "chunk_id" not in chunks or "doc_id" not in chunks:
            raise ValueError("Chunks data must contain chunk_id and doc_id")
        text_columns = [c for c in ("chunk_text", "text", "content", "body", "abstract") if c in chunks]
        text_column = text_columns[0] if text_columns else None
        clean = chunks[["chunk_id", "doc_id"] + ([text_column] if text_column else [])].dropna(
            subset=["chunk_id", "doc_id"]
        )
        self._chunk_to_doc_id = dict(
            zip(clean["chunk_id"].astype(str), clean["doc_id"].astype(str), strict=False)
        )
        if text_column is not None:
            self._chunk_to_text = dict(
                zip(clean["chunk_id"].astype(str), clean[text_column].fillna("").astype(str), strict=False)
            )

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def sources(self) -> list[str]:
        return sorted(self.frame["source"].unique().tolist())

    @property
    def duplicate_doc_ids(self) -> int:
        return int(self.frame["doc_id"].duplicated(keep=False).sum())

    def document(self, row_index: int) -> Document:
        row = self.frame.iloc[int(row_index)]
        doc_id = str(row["doc_id"])
        return Document(
            row_index=int(row_index),
            doc_id=doc_id,
            source=str(row["source"]),
            title=str(row["title"]),
            chunk_id=str(row["chunk_id"]) if "chunk_id" in row and pd.notna(row["chunk_id"]) else None,
            char_len=int(row["char_len"]) if "char_len" in row and pd.notna(row["char_len"]) else None,
            n_chunks=int(row["n_chunks"]) if "n_chunks" in row and pd.notna(row["n_chunks"]) else None,
            text=self._content_by_id.get(doc_id),
        )

    def document_by_doc_id(self, doc_id: str) -> Document | None:
        row_index = self._doc_row_by_id.get(str(doc_id))
        if row_index is None:
            return None
        return self.document(row_index)

    def document_for_chunk(self, chunk_id: str | int) -> Document | None:
        key = str(chunk_id)
        doc_id = self._chunk_to_doc_id.get(key)
        if doc_id is not None:
            doc = self.document_by_doc_id(doc_id)
            if doc is None:
                return None
            text = self._chunk_to_text.get(key, doc.text)
            return Document(
                row_index=doc.row_index,
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                chunk_id=key,
                char_len=doc.char_len,
                n_chunks=doc.n_chunks,
                text=text,
            )
        try:
            row_index = int(chunk_id)
        except (TypeError, ValueError):
            return None
        if row_index < 0 or row_index >= len(self.frame):
            return None
        doc = self.document(row_index)
        return Document(
            row_index=doc.row_index,
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            chunk_id=key,
            char_len=doc.char_len,
            n_chunks=doc.n_chunks,
            text=doc.text,
        )

    def chunk_text(self, chunk_id: str | int) -> str | None:
        return self._chunk_to_text.get(str(chunk_id))

    def search_texts(self) -> list[str]:
        if not self._content_by_id:
            return self.frame["title"].tolist()
        return [
            f"{title}\n{self._content_by_id.get(doc_id, '')}".strip()
            for doc_id, title in zip(self.frame["doc_id"], self.frame["title"], strict=False)
        ]

    def payload(self, row_index: int) -> dict[str, str | int]:
        doc = self.document(row_index)
        payload: dict[str, str | int] = {
            "row_index": doc.row_index,
            "doc_id": doc.doc_id,
            "source": doc.source,
            "title": doc.title,
        }
        if doc.char_len is not None:
            payload["char_len"] = doc.char_len
        if doc.n_chunks is not None:
            payload["n_chunks"] = doc.n_chunks
        return payload

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(len(self.frame)).encode())
        for value in self.frame["doc_id"].astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
