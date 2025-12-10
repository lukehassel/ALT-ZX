"""
Working implementation to extract fidelity values from quantum circuit equivalence checking.

This extends the standard mqt.yaqs equivalence checker to return the actual fidelity value,
not just a boolean result.
"""

from qiskit.converters import circuit_to_dag
import numpy as np
from mqt.yaqs.digital.equivalence_checker import MPO, iterate


def get_fidelity(circuit1, circuit2, threshold=1e-13, fidelity_threshold=1-1e-13):
    assert circuit1.num_qubits == circuit2.num_qubits, "Circuits must have the same number of qubits."

    # Initialize identity MPO
    mpo = MPO()
    mpo.init_identity(circuit1.num_qubits)

    # Convert to DAG and iterate
    dag1 = circuit_to_dag(circuit1)
    dag2 = circuit_to_dag(circuit2)
    # Iteratively apply layers of gates from two DAGCircuits to an MPO until no gates remain
    iterate(mpo, dag1, dag2, threshold)

    # Extract fidelity from matrix representation
    matrix = mpo.to_matrix()
    trace = np.trace(matrix)
    dimension = 2 ** circuit1.num_qubits

    # Exact fidelity (what we calculate)
    fidelity_exact = abs(trace) / dimension

    # Rounded fidelity (what the library uses internally)
    # The library rounds trace to 1 decimal place before computing fidelity
    trace_rounded = np.round(abs(trace), 1)
    fidelity_rounded = trace_rounded / dimension

    # Check equivalence using library's method (which uses rounded fidelity)
    equivalent = mpo.check_if_identity(fidelity_threshold)

    return {
        'equivalent': equivalent,
        'fidelity': float(fidelity_exact),
        'fidelity_rounded': float(fidelity_rounded)
    }