"""Run GA-QSVM for every combination of datasets, kernels, qubit counts and seeds.

Defaults reproduce the original code with the paper's default configuration
(ncircuit=16, p=0.1, d=5n, nCX=2n). Each run is saved to
``results/ga/<dataset>-<kernel>-n<qubits>-seed<seed>/``.

Parallelism: with ``--jobs 1`` (default) runs go one after another and each
evaluates its circuits in ``--workers`` processes. With ``--jobs N`` up to N
whole runs go at once, one process each, which uses many cores better.

Usage:
    uv run python experiments/run_ga.py --dataset digits --kernel fqk --qubits 5 --seed 0
    uv run python experiments/run_ga.py --dataset digits wine cancer --kernel fqk pqk \\
        --qubits 3 4 5 6 7 --seed 0 --jobs 30
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from ga_qsvm.data import PAPER_SPLITS, paper_split
from ga_qsvm.ga import GAConfig, GATask, run_ga, run_tasks
from ga_qsvm.kernels import FQK_BACKENDS, KERNELS, QSVMFitness

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", required=True, choices=list(PAPER_SPLITS))
    parser.add_argument("--kernel", nargs="+", default=["fqk"], choices=KERNELS)
    parser.add_argument("--qubits", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[None], help="one run per seed")
    parser.add_argument("--depth", type=int, default=None, help="gate budget per qubit d (default 5n)")
    parser.add_argument("--num-cnot", type=int, default=None, help="number of CX gates (default 2n)")
    parser.add_argument("--num-circuit", type=int, default=16)
    parser.add_argument("--num-generation", type=int, default=100)
    parser.add_argument("--prob-mutate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=None, help="cap QSVM solver iterations during the GA")
    parser.add_argument(
        "--fqk-backend",
        choices=FQK_BACKENDS,
        default="statevector",
        help="statevector: same kernel values as Qiskit's FidelityQuantumKernel, ~100x faster",
    )
    parser.add_argument("--jobs", type=int, default=1, help="whole GA runs at the same time")
    parser.add_argument("--serial", action="store_true", help="with --jobs 1: evaluate circuits in one process")
    parser.add_argument("--workers", type=int, default=None, help="with --jobs 1: processes per run")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "ga")

    variants = parser.add_argument_group("variants (off by default; original behaviour otherwise)")
    variants.add_argument("--rotations", type=int, nargs=3, metavar=("RX", "RY", "RZ"), default=None)
    variants.add_argument("--rotation-mode", choices=["legacy", "uniform"], default="legacy")
    variants.add_argument("--fill-all-slots", action="store_true")
    variants.add_argument("--crossover-mode", choices=["rotations", "depth"], default="rotations")
    variants.add_argument("--normalizer-mode", choices=["rotations", "depth_cnot"], default="rotations")
    variants.add_argument("--mutation-distinct", action="store_true")
    variants.add_argument("--keep-elites", action="store_true", help="do not mutate the copied parents")
    return parser


def build_tasks(args) -> list[GATask]:
    tasks = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for dataset, kernel, num_qubits, seed in itertools.product(args.dataset, args.kernel, args.qubits, args.seed):
        config = GAConfig(
            num_qubits=num_qubits,
            depth=args.depth,
            num_cnot=args.num_cnot,
            num_circuit=args.num_circuit,
            num_generation=args.num_generation,
            prob_mutate=args.prob_mutate,
            threshold=args.threshold,
            patience=args.patience,
            seed=seed,
            rotations=args.rotations,
            rotation_mode=args.rotation_mode,
            fill_all_slots=args.fill_all_slots,
            crossover_mode=args.crossover_mode,
            normalizer_mode=args.normalizer_mode,
            mutation_distinct=args.mutation_distinct,
            mutate_elites=not args.keep_elites,
        )
        fitness = QSVMFitness(paper_split(dataset, num_qubits), kernel, args.max_iter, args.fqk_backend)
        name = f"{dataset}-{kernel}-n{num_qubits}-seed{seed if seed is not None else stamp}"
        tasks.append(GATask(name, config, fitness, args.output_dir / name))
    return tasks


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tasks = build_tasks(args)

    if args.jobs > 1:
        print(f"{len(tasks)} runs, {args.jobs} at a time -> {args.output_dir}/")
        for summary in run_tasks(tasks, args.jobs):
            if "error" in summary:
                print(f"{summary['name']}: FAILED {summary['error']}", flush=True)
                continue
            print(
                f"{summary['name']}: best {summary['best_fitness']:.4f} after {summary['generations']} "
                f"generations ({summary['stop_reason']}, {summary['seconds']:.0f} s)",
                flush=True,
            )
        return 0

    for task in tasks:
        print(f"== {task.name} -> {task.output_dir}")
        start = time.perf_counter()
        result = run_ga(task.config, task.fitness, parallel=not args.serial, max_workers=args.workers)
        result.save(task.output_dir)
        print(
            f"best fitness {result.best_fitness:.4f} after {len(result.history)} generations "
            f"({result.stop_reason}, {time.perf_counter() - start:.0f} s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
