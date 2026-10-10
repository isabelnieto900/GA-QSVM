import zlib

import numpy as np
import pytest

from ga_qsvm.data import Split
from ga_qsvm.ga import GAConfig, GATask, run_ga, run_tasks
from ga_qsvm.kernels import QSVMFitness


def structural_score(qc):
    """Deterministic pseudo-fitness in [0, 0.95) that depends on the circuit structure."""
    signature = tuple(
        (inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits)) for inst in qc.data
    )
    return zlib.crc32(repr(signature).encode()) / 2**32 * 0.95


def test_config_defaults_follow_paper():
    config = GAConfig(num_qubits=5)
    assert (config.depth, config.num_cnot, config.num_circuit, config.prob_mutate) == (25, 10, 16, 0.1)
    with pytest.raises(ValueError):
        GAConfig(num_qubits=5, num_circuit=10)


def test_run_ga_is_reproducible_with_seed():
    config = GAConfig(num_qubits=3, num_circuit=8, num_generation=5, seed=3)
    first = run_ga(config, structural_score, parallel=False, verbose=False)
    second = run_ga(config, structural_score, parallel=False, verbose=False)
    assert [h["fitnesses"] for h in first.history] == [h["fitnesses"] for h in second.history]
    assert all(len(h["fitnesses"]) == 8 for h in first.history)
    assert first.best_circuit.num_parameters == 3


def test_best_so_far_is_monotonic_and_matches_result():
    result = run_ga(GAConfig(num_qubits=4, num_circuit=8, num_generation=10, seed=0), structural_score,
                    parallel=False, verbose=False)
    best = [h["best_so_far"] for h in result.history]
    assert best == sorted(best)
    assert result.best_fitness == best[-1]
    assert structural_score(result.best_circuit) == pytest.approx(result.best_fitness)


def test_stops_at_threshold_and_on_patience():
    at_threshold = run_ga(GAConfig(num_qubits=3, num_circuit=4, num_generation=20, seed=0),
                          lambda qc: 1.0, parallel=False, verbose=False)
    assert at_threshold.stop_reason == "threshold" and len(at_threshold.history) == 1

    stale = run_ga(GAConfig(num_qubits=3, num_circuit=4, num_generation=20, patience=2, seed=0),
                   lambda qc: 0.5, parallel=False, verbose=False)
    assert stale.stop_reason == "patience" and len(stale.history) == 4


@pytest.mark.parametrize(
    "variant",
    [
        {"rotation_mode": "uniform"},
        {"rotations": (1, 1, 1)},
        {"fill_all_slots": True},
        {"crossover_mode": "depth"},
        {"normalizer_mode": "depth_cnot", "fill_all_slots": True},
        {"mutation_distinct": True, "mutate_elites": False},
    ],
)
def test_variants_run_and_keep_feature_count(variant):
    config = GAConfig(num_qubits=3, num_circuit=4, num_generation=4, seed=1, **variant)
    result = run_ga(config, structural_score, parallel=False, verbose=False)
    assert result.best_circuit.num_parameters == 3


def test_result_save_writes_artifacts(tmp_path):
    result = run_ga(GAConfig(num_qubits=3, num_circuit=4, num_generation=2, seed=0), structural_score,
                    parallel=False, verbose=False)
    output = result.save(tmp_path / "run")
    assert {p.name for p in output.iterdir()} == {
        "best_circuit.qpy", "best_circuit.txt", "summary.json", "history.csv"
    }


def failing_fitness(qc):
    raise RuntimeError("boom")


def test_run_tasks_matches_serial_runs_and_reports_failures(tmp_path):
    configs = {f"seed{seed}": GAConfig(num_qubits=3, num_circuit=4, num_generation=3, seed=seed) for seed in (0, 1)}
    tasks = [GATask(name, config, structural_score, tmp_path / name) for name, config in configs.items()]
    tasks.append(GATask("broken", GAConfig(num_qubits=3, num_circuit=4, seed=0), failing_fitness, tmp_path / "x"))

    summaries = {summary["name"]: summary for summary in run_tasks(tasks, jobs=2)}

    assert "boom" in summaries["broken"]["error"]
    for name, config in configs.items():
        serial = run_ga(config, structural_score, parallel=False, verbose=False)
        assert summaries[name]["best_fitness"] == serial.best_fitness
        assert (tmp_path / name / "summary.json").exists()


def test_run_tasks_unseeded_runs_differ(tmp_path):
    tasks = [
        GATask(str(i), GAConfig(num_qubits=4, num_circuit=8, num_generation=1), structural_score, tmp_path / str(i))
        for i in range(4)
    ]
    list(run_tasks(tasks, jobs=4))
    circuits = {(tmp_path / str(i) / "best_circuit.txt").read_text() for i in range(4)}
    assert len(circuits) > 1


def test_qsvm_fitness_on_tiny_split():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(16, 3))
    y = (x[:, 0] > 0).astype(int)
    split = Split(x[:10], x[10:], y[:10], y[10:])
    result = run_ga(GAConfig(num_qubits=3, num_circuit=4, num_generation=1, seed=0), structural_score,
                    parallel=False, verbose=False)
    accuracy = QSVMFitness(split, "fqk")(result.best_circuit)
    assert 0.0 <= accuracy <= 1.0
