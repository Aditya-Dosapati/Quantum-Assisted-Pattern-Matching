"""Grover's algorithm components: oracle, diffuser, and search runner."""

import os
import functools
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector


def grover_oracle(qc, marked_state):
    """Apply the Grover oracle for a given marked state bit-string."""
    n = len(marked_state)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)
    qc.h(n - 1)
    if n > 1:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i, bit in enumerate(marked_state):
        if bit == "0":
            qc.x(i)


def diffuser(qc, n):
    """Apply the Grover diffusion operator."""
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    if n > 1:
        qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


@functools.lru_cache(maxsize=256)
def _build_grover_circuit(n_candidates, best_classical):
    """Build the Grover circuit once for a given search space and marked index."""
    n_qubits = max(1, int(np.ceil(np.log2(n_candidates))))
    marked_state = format(best_classical, f"0{n_qubits}b")

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_candidates))))
    for _ in range(iterations):
        grover_oracle(qc, marked_state)
        diffuser(qc, n_qubits)
    return qc, n_qubits, marked_state, iterations


def run_grover_search(n_candidates, best_classical, shots=1024):
    """
    Build and execute a Grover circuit targeting the best classical candidate.

    Returns
    -------
    grover_index : int
        The index amplified by Grover's algorithm.
    counts : dict
        Raw measurement counts.
    qc : QuantumCircuit
        The circuit that was executed.
    n_qubits : int
    marked_state : str
    iterations : int
    """
    qc_base, n_qubits, marked_state, iterations = _build_grover_circuit(n_candidates, best_classical)
    fast_mode = os.getenv("QUANTUM_FAST_MODE", "false").lower() in {"1", "true", "yes", "on"}

    if fast_mode and n_qubits <= 8:
        # Fast path for deployments: use exact statevector probabilities instead of qasm transpilation.
        sv = Statevector.from_instruction(qc_base)
        probs = sv.probabilities_dict()
        counts = {}
        for state, prob in probs.items():
            cnt = int(round(float(prob) * shots))
            if cnt > 0:
                counts[state] = cnt
        if not counts:
            counts[format(best_classical, f"0{n_qubits}b")] = shots
        qc = qc_base.copy()
        qc.measure_all()
    else:
        qc = qc_base.copy()
        qc.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        result = backend.run(transpile(qc, backend), shots=shots).result()
        counts = result.get_counts()

    best_state = max(counts, key=counts.get)
    grover_index = int(best_state, 2)
    if grover_index >= n_candidates:
        grover_index = best_classical

    return grover_index, counts, qc, n_qubits, marked_state, iterations
