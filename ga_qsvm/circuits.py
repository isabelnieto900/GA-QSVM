"""Circuit genome and genetic operators of GA-QSVM (Section 4.1).

Ported from ``qoop.evolution`` (generator.by_num_rotations_and_cnot,
divider/normalizer.by_num_rotation_gate, crossover.onepoint,
mutate.bitflip_mutate_with_normalizer, backend.utilities.compose_circuit).
With default options every function consumes Python's ``random`` in the same
order as the original code, so seeded runs produce identical circuits.

A circuit has exactly ``num_qubits`` parametrized rotations: rotation ``k``
receives feature ``k`` (angle encoding). Every operator re-establishes that
invariant through a normalizer.
"""

from __future__ import annotations

import random

import qiskit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CXGate, HGate, RXGate, RYGate, RZGate

# Order matters: mutation samples from this list with random.choice.
GATE_POOL = (HGate, RXGate, RYGate, RZGate, CXGate)
ROTATIONS = (RXGate, RYGate, RZGate)
ROTATION_NAMES = ("rx", "ry", "rz")


def _num_qubits(gate) -> int:
    return 2 if gate is CXGate else 1


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


def _random_partition(total: int, smallest: int, parts: int) -> list[int]:
    """Verbatim port of qoop's ``generate_random_array`` (same random calls)."""
    remaining = total - parts * smallest
    if remaining < 0:
        raise ValueError("Not enough gate slots for the requested gate counts.")
    counts = [smallest] * parts
    for i in range(parts):
        counts[i] += random.randint(0, remaining)
        remaining -= counts[i] - smallest
        if remaining <= 0:
            break
    while sum(counts) < total:
        counts[random.randint(0, parts - 1)] += 1
    random.shuffle(counts)
    return counts


def rotation_allocation(num_qubits: int, mode: str = "legacy", fixed=None) -> tuple[int, int, int]:
    """Number of (RX, RY, RZ) gates, summing to ``num_qubits``.

    - ``fixed``: use the given triple (nRx, nRy, nRz of Eq. 12).
    - ``"legacy"``: original sampling, biased towards RX.
    - ``"uniform"``: uniform over all triples summing to ``num_qubits``.
    """
    if fixed is not None:
        if sum(fixed) != num_qubits:
            raise ValueError(f"Rotation counts {fixed} must sum to num_qubits={num_qubits}")
        return tuple(fixed)
    if mode == "legacy":
        num_rx = random.randint(0, num_qubits)
        num_ry = random.randint(0, num_qubits - num_rx)
        return num_rx, num_ry, num_qubits - num_rx - num_ry
    if mode == "uniform":
        return random.choice(
            [(x, y, num_qubits - x - y) for x in range(num_qubits + 1) for y in range(num_qubits + 1 - x)]
        )
    raise ValueError(f"Unknown rotation allocation mode: {mode!r}")


def random_circuit(
    num_qubits: int,
    depth: int,
    num_cnot: int,
    *,
    rotation_mode: str = "legacy",
    rotations=None,
    fill_all_slots: bool = False,
) -> qiskit.QuantumCircuit:
    """Random feature map with ``num_qubits`` rotations, ``num_cnot`` CX and H gates.

    ``depth * num_qubits`` gate slots are budgeted. The original code only fills
    one of four random partitions of the free slots with H gates, so circuits
    have far fewer gates than the budget; ``fill_all_slots=True`` fills them all.
    ``depth`` is therefore a gate budget per qubit, not the circuit depth.
    """
    num_rx, num_ry, num_rz = rotation_allocation(num_qubits, rotation_mode, rotations)
    num_other = depth * num_qubits - (num_qubits + num_cnot)
    if num_other < 0:
        raise ValueError("The total number of specified gates exceeds depth * num_qubits.")
    pool = [RXGate] * num_rx + [RYGate] * num_ry + [RZGate] * num_rz + [CXGate] * num_cnot
    smallest = max(1, int(0.15 * num_other))
    partition = _random_partition(num_other, smallest, 4)
    pool += [HGate] * (num_other if fill_all_slots else partition[0])
    random.shuffle(pool)

    qc = qiskit.QuantumCircuit(num_qubits)
    rotation_index = 0
    while pool:
        qubits = list(range(num_qubits))
        random.shuffle(qubits)
        gate = pool.pop()
        operands = qubits[: _num_qubits(gate)]
        if gate in ROTATIONS:
            qc.append(gate(Parameter(f"theta({rotation_index})")), operands)
            rotation_index += 1
        else:
            qc.append(gate(), operands)
    return qc


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def compose(circuits) -> qiskit.QuantumCircuit:
    """Concatenate circuits and rename all parameters to ``theta[0..k-1]`` in gate order.

    Like the original, it overwrites the parameters of the input gates in place.
    """
    qc = qiskit.QuantumCircuit(circuits[0].num_qubits)
    thetas = ParameterVector("theta", sum(len(c.parameters) for c in circuits))
    index = 0
    for circuit in circuits:
        for instruction in circuit:
            if len(instruction.operation.params) == 1:
                instruction.operation.params[0] = thetas[index]
                index += 1
            qc.append(instruction.operation, instruction.qubits)
    return qc


def split_after_rotation(qc: qiskit.QuantumCircuit, k: int):
    """Split right after the ``k``-th rotation gate (qoop ``divider.by_num_rotation_gate``)."""
    first = qiskit.QuantumCircuit(qc.num_qubits)
    second = qiskit.QuantumCircuit(qc.num_qubits)
    count = 0
    for i, instruction in enumerate(qc):
        if instruction.operation.name in ROTATION_NAMES:
            count += 1
        first.append(instruction.operation, instruction.qubits)
        if count == k and i + 1 < len(qc):
            for rest in qc[i + 1 :]:
                second.append(rest.operation, rest.qubits)
            return first, second
    return first, second


def split_at_depth(qc: qiskit.QuantumCircuit, depth: int):
    """Split into the gates in layers ``1..depth`` and the rest (ASAP layering)."""
    first = qiskit.QuantumCircuit(qc.num_qubits)
    second = qiskit.QuantumCircuit(qc.num_qubits)
    qubit_layer = [0] * qc.num_qubits
    for instruction in qc:
        indices = [qc.find_bit(q).index for q in instruction.qubits]
        layer = max(qubit_layer[i] for i in indices) + 1
        for i in indices:
            qubit_layer[i] = layer
        target = first if layer <= depth else second
        target.append(instruction.operation, instruction.qubits)
    return first, second


# --------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------


def normalize_rotations(qc: qiskit.QuantumCircuit, num_rotations: int) -> qiskit.QuantumCircuit:
    """Force exactly ``num_rotations`` parameters: truncate after the last allowed
    rotation, or append RX gates (qoop ``normalizer.by_num_rotation_gate``)."""
    count = len(qc.parameters)
    if count == num_rotations:
        return qc
    if count < num_rotations:
        etas = ParameterVector("eta", num_rotations - count)
        for j, i in enumerate(range(count, num_rotations)):
            qc.rx(etas[j], i % qc.num_qubits)
        return qc
    first, _ = split_after_rotation(qc, num_rotations)
    return first


def normalize_depth_and_cnot(
    qc: qiskit.QuantumCircuit, num_rotations: int, depth: int, num_cnot: int
) -> qiskit.QuantumCircuit:
    """Normalizer as described in the paper text: truncate to circuit depth ``depth``,
    append CX gates on random qubit pairs until ``num_cnot`` are present, then
    enforce ``num_rotations`` parameters. The final depth can exceed ``depth``
    by the gates appended in the last two steps."""
    if qc.depth() > depth:
        qc, _ = split_at_depth(qc, depth)
    missing = num_cnot - sum(1 for inst in qc if inst.operation.name == "cx")
    for _ in range(max(0, missing)):
        control, target = random.sample(range(qc.num_qubits), 2)
        qc.cx(control, target)
    return normalize_rotations(qc, num_rotations)


# --------------------------------------------------------------------------
# Crossover and mutation
# --------------------------------------------------------------------------


def crossover(parent1, parent2, *, num_rotations: int, mode: str = "rotations", normalizer):
    """One-point crossover producing two offspring.

    - ``"rotations"`` (original): cut each parent after its ``num_rotations // 2``-th rotation.
    - ``"depth"`` (paper Fig. 2b): cut both parents at half the depth of ``parent1``.
    """
    if mode == "rotations":
        cut = lambda qc: split_after_rotation(qc, num_rotations // 2)
    elif mode == "depth":
        half = parent1.depth() // 2
        cut = lambda qc: split_at_depth(qc, half)
    else:
        raise ValueError(f"Unknown crossover mode: {mode!r}")
    head1, tail1 = cut(parent1)
    head2, tail2 = cut(parent2)
    return normalizer(compose([head1, tail2])), normalizer(compose([head2, tail1]))


def _replace_gate(qc: qiskit.QuantumCircuit, index: int, distinct: bool) -> qiskit.QuantumCircuit:
    """Replace gate ``index`` by a random pool gate acting on the same number of qubits."""
    current = qc.data[index].operation
    while True:
        gate_class = random.choice(GATE_POOL)
        gate = gate_class(Parameter(f"{index}")) if gate_class in ROTATIONS else gate_class()
        if gate.num_qubits != current.num_qubits:
            continue
        # CX is the only two-qubit gate, so it can never change into a distinct gate.
        if distinct and gate.num_qubits == 1 and gate.name == current.name:
            continue
        break
    qubits = qc.data[index].qubits
    qc.data[index] = (gate, list(qubits[: gate.num_qubits]), [])
    return qc


def mutate(qc: qiskit.QuantumCircuit, prob_mutate: float, *, normalizer, distinct: bool = False):
    """Bit-flip mutation: each gate is replaced with probability ``prob_mutate``."""
    for index in range(len(qc.data)):
        if random.random() < prob_mutate:
            qc = _replace_gate(qc, index, distinct)
    return normalizer(qc)


def gate_counts(qc: qiskit.QuantumCircuit) -> dict[str, int]:
    counts = {name: 0 for name in ("h", "rx", "ry", "rz", "cx")}
    for name, value in qc.count_ops().items():
        counts[name] = counts.get(name, 0) + value
    return counts


__all__ = [
    "GATE_POOL",
    "compose",
    "crossover",
    "gate_counts",
    "mutate",
    "normalize_depth_and_cnot",
    "normalize_rotations",
    "random_circuit",
    "rotation_allocation",
    "split_after_rotation",
    "split_at_depth",
]
