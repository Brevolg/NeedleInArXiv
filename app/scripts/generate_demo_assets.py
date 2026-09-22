from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.encoders import HashingEncoder
from search.splade import SPLADEIndex, create_sparse_encoder


DOCUMENTS = [
    ("demo_001", "github", "Multipart file upload limits for API requests"),
    ("demo_002", "github", "Streaming session timebox metrics and finalization"),
    ("demo_003", "linear", "Accessible interactive color tokens for data grids"),
    ("demo_004", "gdrive", "Disaster recovery plan for European region outage"),
    ("demo_005", "fireflies", "Marketplace entitlement propagation during onboarding"),
    ("demo_006", "confluence", "Hybrid retrieval with dense embeddings and BM25"),
    ("demo_007", "jira", "HNSW index latency and recall benchmark"),
    ("demo_008", "gmail", "Security review for customer data retention policy"),
    ("demo_009", "slack", "Incident response notes for API gateway errors"),
    ("demo_010", "hubspot", "Enterprise account renewal and expansion brief"),
]


def main() -> None:
    data_dir = Path("data/demo")
    embedding_dir = Path("embeddings/demo")
    splade_dir = Path("index/demo_splade")
    data_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(DOCUMENTS, columns=["doc_id", "source", "title"])
    mapping["char_len"] = mapping["title"].str.len()
    mapping["n_chunks"] = 1
    mapping.to_parquet(data_dir / "id_mapping.parquet", index=False)
    questions = pd.DataFrame(
        [
            {
                "question_id": "demo_q1",
                "question": "What limits apply to multipart file uploads?",
                "expected_doc_ids": np.array(["demo_001"]),
            },
            {
                "question_id": "demo_q2",
                "question": "How do we recover from a European cloud outage?",
                "expected_doc_ids": np.array(["demo_004"]),
            },
        ]
    )
    questions.to_parquet(data_dir / "questions.parquet", index=False)
    encoder = HashingEncoder(64)
    vectors = encoder.encode(mapping["title"].tolist())
    np.save(embedding_dir / "embeddings_v1.npy", vectors, allow_pickle=False)
    sparse_encoder = create_sparse_encoder("hashing_sparse:256")
    sparse_matrix, order = sparse_encoder.encode(mapping["title"].tolist(), sort_by_length=True)
    SPLADEIndex(
        sparse_matrix,
        row_indices=np.arange(len(mapping), dtype=np.int64)[order],
        sources=mapping["source"].astype(str).to_numpy()[order],
        metadata={
            "version": 1,
            "rows": int(sparse_matrix.shape[0]),
            "features": int(sparse_matrix.shape[1]),
            "model": "hashing_sparse:256",
            "fingerprint": "",
            "content_kind": "title",
            "ordering": "demo hashing index; matrix row i maps to row_indices[i]",
        },
    ).save(splade_dir)
    config = {
        "DATA_PATH": str(data_dir / "id_mapping.parquet"),
        "QUESTIONS_PATH": str(data_dir / "questions.parquet"),
        "EMBEDDINGS_V1_PATH": str(embedding_dir / "embeddings_v1.npy"),
        "MODEL_V1": "hashing:64",
        "QDRANT_COLLECTION_V1": "demo_papers_v1",
        "QDRANT_URL": "file://index/demo_qdrant",
        "BM25_DIR": "index/demo_bm25",
        "SPLADE_DIR": str(splade_dir),
        "SPLADE_MODEL": "hashing_sparse:256",
        "DEFAULT_MODE": "triple_hybrid_v1",
    }
    Path(".env.demo").write_text(
        "\n".join(f"{key}={value}" for key, value in config.items()) + "\n", "utf-8"
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
