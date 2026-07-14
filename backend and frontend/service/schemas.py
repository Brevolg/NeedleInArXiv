from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from search.engine import SearchMode


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: SearchMode = SearchMode.HYBRID_V1
    top_k: int = Field(default=10, ge=1, le=100)
    sources: list[str] | None = None
    rerank: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class ResultItem(BaseModel):
    rank: int
    row_index: int
    doc_id: str
    chunk_id: str | None = None
    title: str
    source: str
    score: float
    search_mode: str
    char_len: int | None = None
    n_chunks: int | None = None
    snippet: str | None = None


class SearchResponseModel(BaseModel):
    query: str
    mode: str
    latency_ms: float
    total: int
    results: list[ResultItem]
