"""Quantum kernels and the GA fitness function (Algorithm 1, Eq. 8)."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from ga_qsvm.data import Split

KERNELS = ("fqk", "pqk")
FQK_BACKENDS = ("statevector", "qiskit")


class StatevectorFidelityKernel:
    """K(x, y) = |<psi(x)|psi(y)>|^2 from one statevector per sample.

    Same values as Qiskit's default ``FidelityQuantumKernel`` (exact Sampler),
    including its defaults for the training matrix: unit diagonal and projection
    onto the closest PSD matrix. Qiskit simulates one circuit per pair of samples;
    this simulates one per sample, which is orders of magnitude faster.
    """

    def __init__(self, circuit):
        self.circuit = circuit

    def statevectors(self, x: np.ndarray) -> np.ndarray:
        parameters = self.circuit.parameters
        return np.array(
            [Statevector(self.circuit.assign_parameters(dict(zip(parameters, row)))).data for row in x]
        )

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        states_x = self.statevectors(x)
        if not np.array_equal(x, y):
            return np.abs(states_x.conj() @ self.statevectors(y).T) ** 2
        kernel = np.abs(states_x.conj() @ states_x.T) ** 2
        np.fill_diagonal(kernel, 1.0)
        w, v = np.linalg.eig(kernel)
        return (v @ np.diag(np.maximum(0, w)) @ v.transpose()).real


def make_qsvc(circuit, kernel: str, max_iter: int | None = None, fqk_backend: str = "statevector"):
    """QSVC with a fidelity or projected (squlearn) quantum kernel on ``circuit``.

    ``max_iter`` caps the SVM solver iterations (the paper runs the fitness with
    fewer iterations during the GA); ``None`` means no limit, as in the original code.
    ``fqk_backend="qiskit"`` uses Qiskit's ``FidelityQuantumKernel`` like the original code.
    """
    svc_kwargs = {} if max_iter is None else {"max_iter": max_iter}
    if kernel == "fqk":
        if fqk_backend == "statevector":
            return SVC(kernel=StatevectorFidelityKernel(circuit), **svc_kwargs)
        if fqk_backend == "qiskit":
            return QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=circuit), **svc_kwargs)
        raise ValueError(f"Unknown FQK backend {fqk_backend!r}; expected one of {FQK_BACKENDS}")
    if kernel == "pqk":
        from squlearn import Executor
        from squlearn.encoding_circuit import QiskitEncodingCircuit
        from squlearn.kernel import QSVC as ProjectedQSVC
        from squlearn.kernel import ProjectedQuantumKernel

        encoding_circuit = QiskitEncodingCircuit(circuit, mode="features")
        quantum_kernel = ProjectedQuantumKernel(
            encoding_circuit=encoding_circuit,
            executor=Executor(),
            initial_parameters=np.random.rand(encoding_circuit.num_parameters),
        )
        return ProjectedQSVC(quantum_kernel=quantum_kernel, **svc_kwargs)
    raise ValueError(f"Unknown kernel {kernel!r}; expected one of {KERNELS}")


class QSVMFitness:
    """Train a QSVM with ``circuit`` as feature map on the training split and
    return the test accuracy. Picklable, so the GA can evaluate it in parallel."""

    def __init__(
        self, split: Split, kernel: str = "fqk", max_iter: int | None = None, fqk_backend: str = "statevector"
    ):
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}; expected one of {KERNELS}")
        self.split = split
        self.kernel = kernel
        self.max_iter = max_iter
        self.fqk_backend = fqk_backend

    def __call__(self, circuit) -> float:
        model = make_qsvc(circuit, self.kernel, self.max_iter, self.fqk_backend)
        model.fit(self.split.x_train, self.split.y_train)
        return float(accuracy_score(self.split.y_test, model.predict(self.split.x_test)))
