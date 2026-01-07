import torch
import sys
sys.path.insert(0, "mpo")
from mpo.gpu.mpo_gpu import get_fidelity_gpu
from qiskit.circuit.random import random_circuit
from qiskit import transpile

device = "cuda:1"
print(f"Testing on {torch.cuda.get_device_name(device)}")

basis = ["cx", "h", "rz", "sx", "x", "id"]

print("\n=== 4-qubit test (identical circuits, should be 1.0) ===")
c1 = random_circuit(4, 5, max_operands=2, seed=42)
c2 = c1.copy()
c1t = transpile(c1, basis_gates=basis, optimization_level=0)
c2t = transpile(c2, basis_gates=basis, optimization_level=0)
result = get_fidelity_gpu(c1t, c2t, device=device, use_randomized_svd=False)
print(f"Fidelity (4q): {result['fidelity']:.10f}")

print("\n=== 8-qubit test (identical circuits, should be 1.0) ===")
c1 = random_circuit(8, 10, max_operands=2, seed=42)
c2 = c1.copy()
c1t = transpile(c1, basis_gates=basis, optimization_level=0)
c2t = transpile(c2, basis_gates=basis, optimization_level=0)
result = get_fidelity_gpu(c1t, c2t, device=device, use_randomized_svd=False)
print(f"Fidelity (8q): {result['fidelity']:.10f}")

print("\n=== 12-qubit test (identical circuits, should be 1.0) ===")
c1 = random_circuit(12, 10, max_operands=2, seed=42)
c2 = c1.copy()
c1t = transpile(c1, basis_gates=basis, optimization_level=0)
c2t = transpile(c2, basis_gates=basis, optimization_level=0)
result = get_fidelity_gpu(c1t, c2t, device=device, use_randomized_svd=True)
print(f"Fidelity (12q): {result['fidelity']:.10f}")
