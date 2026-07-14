from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.encoders import create_encoder
from search.qdrant_store import QdrantStore


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Compare exact scan with HNSW")
    parser.add_argument("--questions", type=Path, default=settings.questions_path)
    parser.add_argument("--collection", default=settings.qdrant_collection_v2)
    parser.add_argument("--model", default=settings.model_v2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-values", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/index_benchmark.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    questions = pd.read_parquet(args.questions).head(args.limit)
    encoder = create_encoder(args.model, settings.device, settings.trust_remote_code)
    vectors = encoder.encode(questions["question"].tolist())
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key)

    exact_results: list[set[str]] = []
    exact_latencies: list[float] = []
    for vector in vectors:
        started = time.perf_counter()
        hits = store.search(args.collection, vector, limit=args.top_k, exact=True)
        exact_latencies.append((time.perf_counter() - started) * 1000)
        exact_results.append({hit.doc_id for hit in hits})

    report: dict = {
        "queries": len(vectors),
        "top_k": args.top_k,
        "exact_latency_mean_ms": fmean(exact_latencies),
        "hnsw": {},
    }
    for ef in args.ef_values:
        latencies: list[float] = []
        overlaps: list[float] = []
        for vector, exact_ids in zip(vectors, exact_results, strict=True):
            started = time.perf_counter()
            hits = store.search(args.collection, vector, limit=args.top_k, exact=False, hnsw_ef=ef)
            latencies.append((time.perf_counter() - started) * 1000)
            approximate_ids = {hit.doc_id for hit in hits}
            overlaps.append(len(exact_ids & approximate_ids) / args.top_k)
        report["hnsw"][str(ef)] = {
            "latency_mean_ms": fmean(latencies),
            "recall_against_exact_at_k": fmean(overlaps),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
