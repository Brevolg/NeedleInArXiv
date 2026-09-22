from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.config import get_settings
from search.engine import SearchEngine, SearchMode
from search.metrics import EvaluationAccumulator


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate retrieval against expected_doc_ids")
    parser.add_argument("--questions", type=Path, default=settings.questions_path)
    parser.add_argument(
        "--modes", nargs="+", choices=[m.value for m in SearchMode], default=["dense_v1"]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = pd.read_parquet(args.questions)
    if args.limit:
        questions = questions.head(args.limit)
    engine = SearchEngine(get_settings())
    output: dict[str, dict] = {}
    for mode in args.modes:
        accumulator = EvaluationAccumulator.create()
        rows: list[dict] = []
        for item in questions.itertuples(index=False):
            response = engine.search(item.question, mode=mode, top_k=args.top_k)
            ranked = [result.doc_id for result in response.results]
            relevant = {str(doc_id) for doc_id in item.expected_doc_ids}
            accumulator.add(ranked, relevant, response.latency_ms, args.top_k)
            rows.append(
                {
                    "question_id": item.question_id,
                    "mode": mode,
                    "latency_ms": response.latency_ms,
                    "retrieved_doc_ids": ranked,
                    "expected_doc_ids": sorted(relevant),
                }
            )
        output[mode] = accumulator.summary()
        detail_path = args.output.with_name(f"{args.output.stem}_{mode}_details.jsonl")
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", "utf-8"
        )
        print(mode, json.dumps(output[mode], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
