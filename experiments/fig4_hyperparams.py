"""Figure 4: GA hyperparameter study on Digits with n = 5 qubits.

Each panel varies one hyperparameter and keeps the others at the base values:
(a) depth d, (b) population size ncircuit, (c) number of CX gates nCX,
(d) mutation probability p. Curves show the best fitness of each generation,
mean and standard deviation over the repeated runs.

Two base configurations:

- ``original`` (default): what the authors' ``benchmark.py`` ran for the paper
  (Sep 2025): PQK, d = 35, nCX = 14, ncircuit = 16, p = 0.1, 200 generations,
  10 runs per configuration, no early stopping.
- ``caption``: the values in the Figure 4 caption: d = 5n = 25, nCX = 2n = 10.

All GA runs are independent and run in parallel, one process per run
(``--jobs``, default: all cores). Each run is saved under
``results/fig4/<base>-<kernel>/runs/`` and skipped if it already exists, so the
study can be stopped and resumed, and the figure can be redrawn with
``--plot-only``.

Usage:
    uv run python experiments/fig4_hyperparams.py --base original caption
    uv run python experiments/fig4_hyperparams.py --repeats 3 --num-generation 100
    uv run python experiments/fig4_hyperparams.py --sweeps num_circuit --repeats 1
    uv run python experiments/fig4_hyperparams.py --base original caption --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ga_qsvm.data import paper_split
from ga_qsvm.ga import GAConfig, GATask, run_tasks
from ga_qsvm.kernels import FQK_BACKENDS, KERNELS, QSVMFitness

REPO_ROOT = Path(__file__).resolve().parents[1]
FONTSIZE = 23
DATASET = "digits"
NUM_QUBITS = 5

SWEEPS = {
    "depth": [5, 10, 15, 20, 25],
    "num_circuit": [4, 8, 16, 20],
    "num_cnot": [5, 10, 15, 20, 25],
    "prob_mutate": [0.001, 0.01, 0.1, 0.3, 0.5],
}
SWEEP_LABELS = {"depth": "d", "num_circuit": "n_{circuit}", "num_cnot": "n_{CX}", "prob_mutate": "p"}
PANEL_LETTERS = dict(zip(SWEEPS, "abcd"))
BASES = {
    "original": {"depth": 35, "num_cnot": 14, "num_circuit": 16, "prob_mutate": 0.1},
    "caption": {"depth": 25, "num_cnot": 10, "num_circuit": 16, "prob_mutate": 0.1},
}


def run_name(params: dict, seed: int) -> str:
    return (
        f"d{params['depth']}-cx{params['num_cnot']}-c{params['num_circuit']}"
        f"-p{params['prob_mutate']:g}-seed{seed}"
    )


def sweep_points(base: dict, sweeps: list[str]):
    """(sweep, value, full parameter set) for every curve of the figure."""
    for sweep in sweeps:
        for value in SWEEPS[sweep]:
            yield sweep, value, {**base, sweep: value}


def runs_dir_for(args, base: str) -> Path:
    output_dir = args.output_dir / f"{base}-{args.kernel}"
    return output_dir / "runs" / f"g{args.num_generation}"


def run_study(args) -> None:
    """Run every missing GA of the selected bases, one process per GA."""
    fitness = QSVMFitness(paper_split(DATASET, NUM_QUBITS), args.kernel, args.max_iter, args.fqk_backend)
    tasks = {}
    for base in args.base:
        runs_dir = runs_dir_for(args, base)
        for _, _, params in sweep_points(BASES[base], args.sweeps):
            for seed in range(args.repeats):
                output = runs_dir / run_name(params, seed)
                config = GAConfig(
                    num_qubits=NUM_QUBITS,
                    num_generation=args.num_generation,
                    patience=args.num_generation,
                    seed=seed,
                    **params,
                )
                tasks.setdefault(output, GATask(f"{base}/{output.name}", config, fitness, output))

    pending = [task for output, task in tasks.items() if not (output / "summary.json").exists()]
    # Longest runs (largest populations) first, so they do not finish last on their own.
    pending.sort(key=lambda task: task.config.num_circuit, reverse=True)
    print(f"{len(tasks)} runs in the study, {len(tasks) - len(pending)} already done, {len(pending)} to run")
    start = time.perf_counter()
    for index, summary in enumerate(run_tasks(pending, args.jobs), start=1):
        elapsed = (time.perf_counter() - start) / 60
        if "error" in summary:
            print(f"[{index}/{len(pending)}] {summary['name']}: FAILED {summary['error']}", flush=True)
            continue
        print(
            f"[{index}/{len(pending)}] {summary['name']}: best {summary['best_fitness']:.3f} "
            f"({summary['seconds'] / 60:.1f} min; {elapsed:.0f} min elapsed)",
            flush=True,
        )


def load_history(run_dir: Path) -> np.ndarray:
    with (run_dir / "history.csv").open() as handle:
        return np.array([float(row["best_fitness"]) for row in csv.DictReader(handle)])


def collect_curves(args, base: str) -> list[dict]:
    """Mean and std of the per-generation best fitness over the available runs."""
    runs_dir = runs_dir_for(args, base)
    rows = []
    for sweep, value, params in sweep_points(BASES[base], args.sweeps):
        histories = [
            load_history(runs_dir / run_name(params, seed))
            for seed in range(args.repeats)
            if (runs_dir / run_name(params, seed) / "history.csv").exists()
        ]
        if not histories:
            continue
        length = max(len(h) for h in histories)
        padded = np.full((len(histories), length), np.nan)
        for i, history in enumerate(histories):
            padded[i, : len(history)] = history
        mean, std = np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)
        for generation in range(length):
            rows.append(
                {
                    "sweep": sweep,
                    "value": value,
                    "generation": generation + 1,
                    "mean_best_fitness": mean[generation],
                    "std_best_fitness": std[generation],
                    "num_runs": len(histories),
                }
            )
    return rows


def plot(rows: list[dict], sweeps: list[str], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.5 * len(sweeps), 6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sweep in zip(axes, sweeps):
        values = SWEEPS[sweep]
        colors = sns.color_palette("viridis_r", len(values))
        for color, value in zip(colors, values):
            curve = [row for row in rows if row["sweep"] == sweep and row["value"] == value]
            if not curve:
                continue
            generation = np.array([row["generation"] for row in curve])
            mean = np.array([row["mean_best_fitness"] for row in curve])
            std = np.array([row["std_best_fitness"] for row in curve])
            ax.plot(generation, mean, linestyle="-", linewidth=2.5, color=color, label=f"${SWEEP_LABELS[sweep]} = {value:g}$")
            if curve[0]["num_runs"] > 1:
                ax.fill_between(generation, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(f"({PANEL_LETTERS[sweep]})", fontsize=FONTSIZE)
        ax.set_xlabel("Generation", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE - 4)
        ax.legend(
            loc="lower right",
            fontsize=FONTSIZE - 8,
            frameon=True,
            fancybox=False,
            facecolor="white",
            edgecolor="lightgray",
        ).get_frame().set_linewidth(0.8)
    axes[0].set_ylabel("Best Fitness", fontsize=FONTSIZE)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig4_hyperparams.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", nargs="+", choices=list(BASES), default=["original"])
    parser.add_argument("--sweeps", nargs="+", choices=list(SWEEPS), default=list(SWEEPS))
    parser.add_argument("--repeats", type=int, default=10, help="GA runs per configuration (seeds 0..repeats-1)")
    parser.add_argument("--num-generation", type=int, default=200)
    parser.add_argument("--kernel", choices=KERNELS, default="pqk")
    parser.add_argument("--max-iter", type=int, default=None, help="cap QSVM solver iterations during the GA")
    parser.add_argument("--fqk-backend", choices=FQK_BACKENDS, default="statevector")
    parser.add_argument("--jobs", type=int, default=None, help="GA runs at the same time (default: all cores)")
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved runs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "fig4")
    args = parser.parse_args(argv)

    for base in args.base:
        runs_dir_for(args, base).mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run_study(args)

    status = 0
    for base in args.base:
        output_dir = args.output_dir / f"{base}-{args.kernel}"
        rows = collect_curves(args, base)
        if not rows:
            print(f"{base}: no finished runs to plot.")
            status = 1
            continue
        study = {**vars(args), "base": base, "base_values": BASES[base]}
        (output_dir / "study.json").write_text(json.dumps(study, indent=2, default=str))
        write_csv(output_dir / "fig4_curves.csv", rows)
        plot(rows, args.sweeps, output_dir)
        print(f"{base}: saved to {output_dir}/")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
