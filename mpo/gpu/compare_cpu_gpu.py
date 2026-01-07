import sys
sys.path.insert(0, "mpo")

from .mpo_gpu import get_fidelity_gpu
from fidelity import get_fidelity
from qiskit.circuit.random import random_circuit
from qiskit import transpile

basis = ["cx", "h", "rz", "sx", "x", "id"]

print("=== Comparing GPU vs CPU fidelity ===")
print()

for n_qubits in [4, 6, 8]:
    print(f"--- {n_qubits} qubits (identical circuits) ---")
    c1 = random_circuit(n_qubits, 5, max_operands=2, seed=42)
    c2 = c1.copy()
    c1t = transpile(c1, basis_gates=basis, optimization_level=0)
    c2t = transpile(c2, basis_gates=basis, optimization_level=0)
    
    cpu_result = get_fidelity(c1t, c2t)
    print(f"  CPU fidelity: {cpu_result['fidelity']:.10f}")
    
    gpu_result = get_fidelity_gpu(c1t, c2t, device="cuda:1", use_randomized_svd=False)
    print(f"  GPU fidelity: {gpu_result['fidelity']:.10f}")
    print()
