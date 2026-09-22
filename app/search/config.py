from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    data_path: Path = Path("data/id_mapping_v1.parquet")
    corpus_path: Path | None = None
    chunks_path: Path | None = None
    questions_path: Path = Path("data/questions.parquet")
    embeddings_v1_path: Path = Path("embeddings/embeddings_v1.npy")
    embeddings_v2_path: Path = Path("embeddings/embeddings_v2.npy")
    bm25_dir: Path = Path("index/bm25")
    splade_dir: Path = Path("index/splade")
    external_indexes_dir: Path = Path("external_indexes")
    use_precomputed_indexes: bool = False
    precomputed_bm25s_dir: Path = Path("external_indexes/bm25s_cache")
    precomputed_dense_dir: Path = Path("external_indexes/dense")
    precomputed_splade_dir: Path = Path("external_indexes/splade")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_v1: str = "papers_v1"
    qdrant_collection_v2: str = "papers_v2"

    model_v1: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_v2: str = "allenai/specter2"
    dense_query_prefix: str = ""
    dense_normalize_query: bool = True
    device: str = "cpu"
    trust_remote_code: bool = False
    splade_model: str = "naver/splade-cocondenser-ensembledistil"
    splade_max_length: int = Field(default=374, ge=16)
    splade_threshold: float = Field(default=0.01, ge=0)
    splade_batch_size: int = Field(default=32, ge=1)
    splade_query_batch_size: int = Field(default=8, ge=1)
    splade_fp16: bool = True

    reranker_enabled: bool = False
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"
    reranker_max_length: int = Field(default=256, ge=16)
    reranker_batch_size: int = Field(default=4, ge=1)
    reranker_candidates: int = Field(default=100, ge=1)
    reranker_trust_remote_code: bool = True

    default_mode: str = "hybrid_v1"
    rrf_k: int = Field(default=60, ge=1)
    hnsw_ef_search: int = Field(default=128, ge=8)
    search_candidates: int = Field(default=50, ge=10)
    max_top_k: int = Field(default=100, ge=1, le=1000)

    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25s_stemmer: str | None = "russian"
    bm25s_stopwords: str | None = "ru"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
