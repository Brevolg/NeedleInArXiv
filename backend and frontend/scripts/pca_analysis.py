from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA analysis on a reproducible embedding sample")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/pca"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectors = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {vectors.shape}")
    sample_size = min(args.sample_size, len(vectors))
    rng = np.random.default_rng(42)
    ids = np.sort(rng.choice(len(vectors), sample_size, replace=False))
    sample = np.asarray(vectors[ids], dtype=np.float32)
    components = min(args.components, sample.shape[0], sample.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=42)
    pca.fit(sample)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    table = pd.DataFrame(
        {
            "component": np.arange(1, components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "pca_components.csv", index=False)
    thresholds = {}
    for threshold in (0.8, 0.9, 0.95):
        reached = np.flatnonzero(cumulative >= threshold)
        thresholds[str(threshold)] = int(reached[0] + 1) if len(reached) else None
    report = {
        "embedding_shape": list(vectors.shape),
        "sample_size": sample_size,
        "components_fitted": components,
        "components_for_thresholds": thresholds,
        "variance_explained_by_fitted_components": float(cumulative[-1]),
    }
    (args.output_dir / "pca_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(table["component"], table["cumulative_explained_variance"], linewidth=2)
        for threshold in (0.8, 0.9, 0.95):
            ax.axhline(threshold, linestyle="--", linewidth=1, label=f"{threshold:.0%}")
        ax.set(xlabel="Components", ylabel="Cumulative explained variance", title="PCA")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "pca_curve.png", dpi=160)
        plt.close(fig)
    except ImportError:
        report["plot"] = "matplotlib is not installed; CSV was generated"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

