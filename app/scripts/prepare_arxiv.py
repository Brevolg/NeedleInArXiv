from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TARGET_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI", "cs.CV", "cs.IR", "stat.ML"}
LATEX_COMMAND = re.compile(r"\\(?:textit|textbf|emph|mathrm|mathbf)\{([^{}]*)\}")
WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    text = str(value).replace("\x00", " ")
    previous = None
    while text != previous:
        previous = text
        text = LATEX_COMMAND.sub(r"\1", text)
    return WHITESPACE.sub(" ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and filter the Kaggle arXiv JSONL dump")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_clean.parquet"))
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--min-documents", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_json(args.input, lines=True)
    required = {"id", "title", "abstract", "categories", "update_date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"arXiv dump is missing columns: {sorted(missing)}")
    raw["year"] = pd.to_datetime(raw["update_date"], errors="coerce").dt.year
    raw["category_list"] = raw["categories"].fillna("").str.split()
    selected = raw[raw["category_list"].map(lambda values: bool(set(values) & TARGET_CATEGORIES))]
    if args.min_year is not None:
        selected = selected[selected["year"] >= args.min_year]
    if args.max_year is not None:
        selected = selected[selected["year"] <= args.max_year]
    selected = selected.copy()
    selected["title"] = selected["title"].map(clean_text)
    selected["abstract"] = selected["abstract"].map(clean_text)
    selected = selected[(selected["title"].str.len() > 0) & (selected["abstract"].str.len() > 20)]
    selected["category"] = selected["category_list"].map(
        lambda values: next((v for v in values if v in TARGET_CATEGORIES), values[0])
    )
    selected = selected.drop_duplicates("id", keep="last").drop_duplicates(
        ["title", "abstract"], keep="first"
    )
    clean = selected.rename(columns={"id": "paper_id"})[
        ["paper_id", "title", "abstract", "category", "year"]
    ].sort_values("paper_id")
    if len(clean) < args.min_documents:
        raise RuntimeError(
            f"The selected slice contains {len(clean):,} documents, below {args.min_documents:,}. "
            "Adjust the year bounds instead of silently accepting an undersized dataset."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.output, index=False)
    report_dir = Path("reports/generated/arxiv")
    report_dir.mkdir(parents=True, exist_ok=True)
    clean["category"].value_counts().rename_axis("category").rename("documents").to_csv(
        report_dir / "category_distribution.csv"
    )
    clean["year"].value_counts().sort_index().rename_axis("year").rename("documents").to_csv(
        report_dir / "year_distribution.csv"
    )
    print(f"Saved {len(clean):,} cleaned articles to {args.output}")


if __name__ == "__main__":
    main()

