import random

import pytest

from ga_qsvm import circuits


def _random_circuit(n=5, seed=0, **kwargs):
    random.seed(seed)
    return circuits.random_circuit(n, 5 * n, 2 * n, **kwargs)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_random_circuit_has_one_rotation_per_qubit_and_required_cnots(n):
    for seed in range(20):
        qc = _random_circuit(n, seed)
        counts = circuits.gate_counts(qc)
        assert qc.num_parameters == n
        assert counts["rx"] + counts["ry"] + counts["rz"] == n
        assert counts["cx"] == 2 * n
        assert len(qc.data) < 5 * n * n


def test_fill_all_slots_uses_full_gate_budget():
    qc = _random_circuit(5, fill_all_slots=True)
    assert len(qc.data) == 5 * 5 * 5


def test_rotation_allocation_modes():
    random.seed(0)
    assert circuits.rotation_allocation(5, fixed=(1, 2, 2)) == (1, 2, 2)
    for mode in ("legacy", "uniform"):
        assert all(sum(circuits.rotation_allocation(5, mode)) == 5 for _ in range(50))
    with pytest.raises(ValueError):
        circuits.rotation_allocation(5, fixed=(1, 1, 1))


def test_normalize_rotations_adds_or_truncates_to_exact_count():
    qc = _random_circuit(5)
    assert circuits.normalize_rotations(qc.copy(), 7).num_parameters == 7
    assert circuits.normalize_rotations(qc.copy(), 3).num_parameters == 3


def test_split_at_depth_keeps_all_gates_and_respects_cut():
    qc = _random_circuit(5, fill_all_slots=True)
    first, second = circuits.split_at_depth(qc, 6)
    assert first.depth() <= 6
    assert len(first.data) + len(second.data) == len(qc.data)


def test_normalize_depth_and_cnot():
    qc = _random_circuit(5, fill_all_slots=True)
    out = circuits.normalize_depth_and_cnot(qc, num_rotations=5, depth=8, num_cnot=10)
    assert out.num_parameters == 5
    assert circuits.gate_counts(out)["cx"] >= min(10, circuits.gate_counts(qc)["cx"])


@pytest.mark.parametrize("mode", ["rotations", "depth"])
def test_crossover_offspring_keep_rotation_count(mode):
    normalizer = lambda qc: circuits.normalize_rotations(qc, 5)
    for seed in range(10):
        a, b = _random_circuit(5, seed), _random_circuit(5, seed + 100)
        children = circuits.crossover(a, b, num_rotations=5, mode=mode, normalizer=normalizer)
        assert all(child.num_parameters == 5 for child in children)


def test_mutation_keeps_rotation_count_and_distinct_changes_gates():
    normalizer = lambda qc: circuits.normalize_rotations(qc, 5)
    random.seed(1)
    qc = _random_circuit(5)
    names_before = [inst.operation.name for inst in qc.data]
    mutated = circuits.mutate(qc.copy(), 1.0, normalizer=lambda c: c, distinct=True)
    names_after = [inst.operation.name for inst in mutated.data]
    for before, after in zip(names_before, names_after):
        assert before == after == "cx" or before != after
    assert circuits.mutate(qc.copy(), 0.5, normalizer=normalizer).num_parameters == 5


def test_compose_renames_parameters_in_gate_order():
    qc = circuits.compose([_random_circuit(4)])
    rotation_params = [inst.operation.params[0] for inst in qc.data if inst.operation.params]
    assert [p.name for p in rotation_params] == [f"theta[{i}]" for i in range(4)]
