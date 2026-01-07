from qiskit.converters import circuit_to_dag
import numpy as np
from mqt.yaqs.digital.equivalence_checker import MPO, iterate


def get_fidelity(circuit1, circuit2, threshold=1e-13, fidelity_threshold=1-1e-13):
    assert circuit1.num_qubits == circuit2.num_qubits, "Circuits must have the same number of qubits."

    num_qubits = circuit1.num_qubits
    dimension = 2 ** num_qubits

    mpo = MPO()
    mpo.init_identity(num_qubits)

    dag1 = circuit_to_dag(circuit1)
    dag2 = circuit_to_dag(circuit2)
    iterate(mpo, dag1, dag2, threshold)

    identity_mpo = MPO()
    identity_mpo.init_identity(num_qubits)
    
    identity_mps = identity_mpo.to_mps()
    mps = mpo.to_mps()
    trace = mps.scalar_product(identity_mps)

    fidelity_exact = abs(trace) / dimension
    
    trace_rounded = np.round(abs(trace), 1)
    fidelity_rounded = trace_rounded / dimension

    equivalent = fidelity_exact >= fidelity_threshold

    return {
        'equivalent': equivalent,
        'fidelity': float(fidelity_exact),
        'fidelity_rounded': float(fidelity_rounded)
    }