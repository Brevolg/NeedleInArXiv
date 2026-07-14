from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.data import DataRepository


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate mapping, qrels, and embeddings")
    parser.add_argument("--mapping", type=Path, default=settings.data_path)
    parser.add_argument("--corpus", type=Path, default=settings.corpus_path)
    parser.add_argument("--questions", type=Path, default=settings.questions_path)
    parser.add_argument("--embeddings", type=Path, default=settings.embeddings_v1_path)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/input_validation.json"))
    parser.add_argument("--require-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DataRepository(args.mapping, args.corpus)
    frame = repository.frame
    report: dict = {
        "status": "ok",
        "mapping": {
            "path": str(args.mapping),
            "rows": len(repository),
            "unique_doc_ids": int(frame["doc_id"].nunique()),
            "duplicate_rows": repository.duplicate_doc_ids,
            "sources": frame["source"].value_counts().sort_index().to_dict(),
            "fingerprint_sha256": repository.fingerprint(),
            "content": repository.content_column or "title_only",
        },
        "warnings": [],
        "errors": [],
    }
    if repository.duplicate_doc_ids:
        report["warnings"].append(
            "Duplicate doc_id rows exist; retain row order and deduplicate only after retrieval."
        )

    if args.questions.exists():
        questions = pd.read_parquet(args.questions)
        required = {"question_id", "question", "expected_doc_ids"}
        missing = required.difference(questions.columns)
        if missing:
            report["errors"].append(f"Questions missing columns: {sorted(missing)}")
        else:
            known = set(frame["doc_id"])
            expected = [str(doc_id) for ids in questions["expected_doc_ids"] for doc_id in ids]
            covered = sum(doc_id in known for doc_id in expected)
            report["questions"] = {
                "path": str(args.questions),
                "rows": len(questions),
                "expected_doc_references": len(expected),
                "covered_references": covered,
                "coverage": covered / len(expected) if expected else None,
                "queries_without_expected_docs": int(
                    sum(len(ids) == 0 for ids in questions["expected_doc_ids"])
                ),
            }
            if covered != len(expected):
                report["errors"].append(
                    f"Only {covered}/{len(expected)} expected document references exist in mapping"
                )
    else:
        report["warnings"].append(f"Questions file is absent: {args.questions}")

    if args.embeddings.exists():
        vectors = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
        report["embeddings"] = {
            "path": str(args.embeddings),
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "size_bytes": args.embeddings.stat().st_size,
        }
        if vectors.ndim != 2:
            report["errors"].append(f"Embeddings are not 2D: {vectors.shape}")
        elif vectors.shape[0] != len(repository):
            report["errors"].append(
                f"Embedding rows {vectors.shape[0]} != mapping rows {len(repository)}"
            )
        if vectors.dtype != np.float32:
            report["warnings"].append(f"Expected float32 embeddings, got {vectors.dtype}")
        sample_rows = min(len(vectors), 10_000)
        if sample_rows and not np.isfinite(np.asarray(vectors[:sample_rows])).all():
            report["errors"].append("Non-finite values found in first 10,000 embedding rows")
    elif args.require_embeddings:
        report["errors"].append(f"Embeddings file is absent: {args.embeddings}")
    else:
        report["warnings"].append(
            f"Embeddings file is absent: {args.embeddings}; code/data checks can still run"
        )

    if report["errors"]:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
