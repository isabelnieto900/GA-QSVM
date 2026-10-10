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

Each GA run is saved under ``results/fig4/runs/`` and skipped if it already
exists, so the study can be stopped and resumed, and the figure can be redrawn
with ``--plot-only``.

Usage:
    uv run python experiments/fig4_hyperparams.py
    uv run python experiments/fig4_hyperparams.py --repeats 3 --num-generation 100
    uv run python experiments/fig4_hyperparams.py --sweeps num_circuit --repeats 1
    uv run python experiments/fig4_hyperparams.py --plot-only
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
from ga_qsvm.ga import GAConfig, run_ga
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


def run_study(args, runs_dir: Path) -> None:
    fitness = QSVMFitness(paper_split(DATASET, NUM_QUBITS), args.kernel, args.max_iter, args.fqk_backend)
    jobs = {}
    for _, _, params in sweep_points(BASES[args.base], args.sweeps):
        for seed in range(args.repeats):
            jobs.setdefault(run_name(params, seed), (params, seed))

    pending = [(name, job) for name, job in jobs.items() if not (runs_dir / name / "summary.json").exists()]
    print(f"{len(jobs)} runs in the study, {len(jobs) - len(pending)} already done, {len(pending)} to run")
    for index, (name, (params, seed)) in enumerate(pending, start=1):
        config = GAConfig(
            num_qubits=NUM_QUBITS,
            num_generation=args.num_generation,
            patience=args.num_generation,
            seed=seed,
            **params,
        )
        start = time.perf_counter()
        result = run_ga(config, fitness, max_workers=args.workers, verbose=False)
        result.save(runs_dir / name)
        print(
            f"[{index}/{len(pending)}] {name}: best {result.best_fitness:.3f} "
            f"({len(result.history)} generations, {time.perf_counter() - start:.0f} s)",
            flush=True,
        )


def load_history(run_dir: Path) -> np.ndarray:
    with (run_dir / "history.csv").open() as handle:
        return np.array([float(row["best_fitness"]) for row in csv.DictReader(handle)])


def collect_curves(args, runs_dir: Path) -> list[dict]:
    """Mean and std of the per-generation best fitness over the available runs."""
    rows = []
    for sweep, value, params in sweep_points(BASES[args.base], args.sweeps):
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
    parser.add_argument("--base", choices=list(BASES), default="original")
    parser.add_argument("--sweeps", nargs="+", choices=list(SWEEPS), default=list(SWEEPS))
    parser.add_argument("--repeats", type=int, default=10, help="GA runs per configuration (seeds 0..repeats-1)")
    parser.add_argument("--num-generation", type=int, default=200)
    parser.add_argument("--kernel", choices=KERNELS, default="pqk")
    parser.add_argument("--max-iter", type=int, default=None, help="cap QSVM solver iterations during the GA")
    parser.add_argument("--fqk-backend", choices=FQK_BACKENDS, default="statevector")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved runs")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/fig4/<base>-<kernel>")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or REPO_ROOT / "results" / "fig4" / f"{args.base}-{args.kernel}"
    runs_dir = output_dir / "runs" / f"g{args.num_generation}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "study.json").write_text(json.dumps({**vars(args), "output_dir": output_dir}, indent=2, default=str))

    if not args.plot_only:
        run_study(args, runs_dir)
    rows = collect_curves(args, runs_dir)
    if not rows:
        print("No finished runs to plot.")
        return 1
    write_csv(output_dir / "fig4_curves.csv", rows)
    plot(rows, args.sweeps, output_dir)
    print(f"Saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
