"""Figure 3: cumulative PCA explained variance of each dataset.

Usage:
    uv run python experiments/fig3_pca.py
    uv run python experiments/fig3_pca.py --datasets digits wine cancer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ga_qsvm.data import DATASET_LABELS, DATASETS, components_for, explained_variance_curve

FONTSIZE = 23
PAPER_COMPONENTS_95 = {"digits": 30, "fashion": 200, "wine": 10, "cancer": 10}


def plot(curves: dict, threshold: float, output_dir: Path) -> None:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 6))
    axes = [axes] if len(curves) == 1 else axes
    for letter, ax, (name, curve) in zip("abcd", axes, curves.items()):
        k = components_for(curve, threshold)
        components = range(1, len(curve) + 1)
        ax.plot(components, curve, linewidth=2.5, color="tab:blue")
        ax.axhline(threshold, linestyle="--", linewidth=1.5, color="tab:red")
        ax.axvline(k, linestyle="--", linewidth=1.5, color="tab:gray")
        ax.annotate(
            f"{k} components",
            xy=(k, threshold),
            xytext=(0.45, 0.25),
            textcoords="axes fraction",
            fontsize=FONTSIZE - 5,
            arrowprops={"arrowstyle": "->", "color": "black"},
        )
        ax.set_title(f"({letter}) {DATASET_LABELS[name]}", fontsize=FONTSIZE)
        ax.set_xlabel("PCA components", fontsize=FONTSIZE)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=FONTSIZE - 4)
        ax.grid(axis="both", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Cumulative explained variance", fontsize=FONTSIZE - 2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig3_pca.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fig3"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = {name: explained_variance_curve(name) for name in args.datasets}

    summary = [
        {
            "dataset": name,
            "n_features": len(curve),
            "components_90": components_for(curve, 0.90),
            "components_95": components_for(curve, 0.95),
            "paper_components_95": PAPER_COMPONENTS_95[name],
        }
        for name, curve in curves.items()
    ]
    curve_rows = [
        {"dataset": name, "component": i, "cumulative_explained_variance": float(value)}
        for name, curve in curves.items()
        for i, value in enumerate(curve, start=1)
    ]
    write_csv(args.output_dir / "pca_components.csv", summary)
    write_csv(args.output_dir / "pca_curves.csv", curve_rows)
    plot(curves, args.threshold, args.output_dir)

    for row in summary:
        status = "OK" if row["components_95"] == row["paper_components_95"] else "DIFFERS"
        print(
            f"{row['dataset']:8s} 95% -> {row['components_95']:4d} components "
            f"(paper: {row['paper_components_95']}) {status}"
        )
    print(f"Saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
