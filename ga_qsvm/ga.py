"""Genetic algorithm of GA-QSVM (Algorithm 2), ported from ``qoop.evolution.environment``.

Default options reproduce the original code. The ``*_mode`` and boolean options
switch on the variants described in the paper text or fix known bugs; see
``GAConfig``.
"""

from __future__ import annotations

import concurrent.futures
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import qiskit
from qiskit import qpy

from ga_qsvm import circuits


@dataclass
class GAConfig:
    """Metadata M of Eq. 12 plus implementation options.

    Defaults follow the paper's default configuration: d = 5n, nCX = 2n,
    ncircuit = 16, p = 0.1. ``depth`` is a gate budget per qubit (see
    ``circuits.random_circuit``), not the circuit depth, unless
    ``normalizer_mode="depth_cnot"``.
    """

    num_qubits: int
    depth: int | None = None
    num_cnot: int | None = None
    num_circuit: int = 16
    num_generation: int = 100
    prob_mutate: float = 0.1
    threshold: float = 0.99
    patience: int = 50
    seed: int | None = None
    # Rotation counts (nRx, nRy, nRz). None samples them per circuit.
    rotations: tuple[int, int, int] | None = None
    rotation_mode: str = "legacy"  # "legacy" (biased to RX) | "uniform"
    fill_all_slots: bool = False  # fix: use the full depth * n gate budget
    crossover_mode: str = "rotations"  # "rotations" (original) | "depth" (paper Fig. 2b)
    normalizer_mode: str = "rotations"  # "rotations" (original) | "depth_cnot" (paper text)
    mutation_distinct: bool = False  # paper: mutated gate must differ from the original
    mutate_elites: bool = True  # original mutates the copied parents too

    def __post_init__(self):
        if self.depth is None:
            self.depth = 5 * self.num_qubits
        if self.num_cnot is None:
            self.num_cnot = 2 * self.num_qubits
        if self.rotations is not None:
            self.rotations = tuple(self.rotations)
        if self.num_circuit % 4 != 0:
            raise ValueError("num_circuit must be divisible by 4 to keep the population size.")

    def normalizer(self, qc):
        if self.normalizer_mode == "rotations":
            return circuits.normalize_rotations(qc, self.num_qubits)
        if self.normalizer_mode == "depth_cnot":
            return circuits.normalize_depth_and_cnot(qc, self.num_qubits, self.depth, self.num_cnot)
        raise ValueError(f"Unknown normalizer mode: {self.normalizer_mode!r}")

    def new_circuit(self):
        return circuits.random_circuit(
            self.num_qubits,
            self.depth,
            self.num_cnot,
            rotation_mode=self.rotation_mode,
            rotations=self.rotations,
            fill_all_slots=self.fill_all_slots,
        )


@dataclass
class GAResult:
    config: GAConfig
    best_circuit: qiskit.QuantumCircuit
    best_fitness: float
    stop_reason: str
    history: list[dict] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "best_circuit.qpy").open("wb") as handle:
            qpy.dump(self.best_circuit, handle)
        (output / "best_circuit.txt").write_text(str(self.best_circuit.draw(output="text", fold=-1)))
        summary = {
            "best_fitness": self.best_fitness,
            "stop_reason": self.stop_reason,
            "generations": len(self.history),
            "best_circuit_gates": circuits.gate_counts(self.best_circuit),
            "best_circuit_depth": self.best_circuit.depth(),
            "config": asdict(self.config),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        with (output / "history.csv").open("w", newline="") as handle:
            fieldnames = ["generation", "best_fitness", "mean_fitness", "best_so_far", "fitnesses"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow({**row, "fitnesses": " ".join(f"{f:.4f}" for f in row["fitnesses"])})
        return output


def initial_population(config: GAConfig) -> list[qiskit.QuantumCircuit]:
    population = []
    while len(population) < config.num_circuit:
        qc = config.new_circuit()
        if len(qc.parameters) > 0:
            population.append(qc)
    return population


def elitist_selection(population, fitnesses):
    """Keep the best half, sorted by fitness (stable for ties)."""
    ranked = sorted(zip(population, fitnesses), key=lambda pair: pair[1], reverse=True)
    return [qc for qc, _ in ranked[: len(population) // 2]]


def evaluate(fitness: Callable, population, parallel: bool, max_workers: int | None = None) -> list[float]:
    if not parallel:
        return [float(fitness(qc)) for qc in population]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        return [float(value) for value in executor.map(fitness, population)]


def next_generation(config: GAConfig, population, fitnesses):
    """Selection, crossover and mutation (Eqs. 9-11)."""
    parents = elitist_selection(population, fitnesses)
    children = []
    for i in range(0, config.num_circuit // 2, 2):
        offspring1, offspring2 = circuits.crossover(
            parents[i],
            parents[i + 1],
            num_rotations=config.num_qubits,
            mode=config.crossover_mode,
            normalizer=config.normalizer,
        )
        children.extend([parents[i].copy(), parents[i + 1].copy(), offspring1, offspring2])
    for i, child in enumerate(children):
        is_elite = i % 4 < 2
        if config.mutate_elites or not is_elite:
            child = circuits.mutate(
                child, config.prob_mutate, normalizer=config.normalizer, distinct=config.mutation_distinct
            )
        children[i] = circuits.compose([child])
    return children


def run_ga(
    config: GAConfig,
    fitness: Callable[[qiskit.QuantumCircuit], float],
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    verbose: bool = True,
) -> GAResult:
    """Evolve circuits until the best fitness exceeds ``threshold``, it does not
    improve for more than ``patience`` generations, or ``num_generation`` is reached."""
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    population = initial_population(config)
    best_fitness, best_circuit = 0.0, None
    stale, stop_reason = 0, "max_generations"
    history = []
    for generation in range(1, config.num_generation + 1):
        fitnesses = evaluate(fitness, population, parallel, max_workers)
        index = int(np.argmax(fitnesses))
        if best_circuit is None:
            best_circuit = copy.deepcopy(population[index])
        improved = fitnesses[index] > best_fitness
        if improved:
            best_fitness = fitnesses[index]
            best_circuit = copy.deepcopy(population[index])
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "generation": generation,
                "best_fitness": fitnesses[index],
                "mean_fitness": float(np.mean(fitnesses)),
                "best_so_far": best_fitness,
                "fitnesses": fitnesses,
            }
        )
        if verbose:
            print(
                f"gen {generation:3d}  best {fitnesses[index]:.4f}  mean {np.mean(fitnesses):.4f}  "
                f"best so far {best_fitness:.4f}",
                flush=True,
            )
        if improved and best_fitness > config.threshold:
            stop_reason = "threshold"
            break
        if stale > config.patience:
            stop_reason = "patience"
            break
        population = next_generation(config, population, fitnesses)

    return GAResult(config, best_circuit, best_fitness, stop_reason, history)
