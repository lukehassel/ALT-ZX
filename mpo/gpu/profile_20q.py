import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from mpo.gpu.mpo_gpu import get_fidelity_gpu


def profile_20q():
    n = 20
    depth = 20
    basis_gates = ['cp', 'cx', 'cz', 'h', 'id', 'p', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzz', 'swap', 'sx', 'x', 'y', 'z']
    
    print(f"Generating random circuit (n={n}, d={depth})...")
    c1 = random_circuit(n, depth, seed=42)
    c2 = random_circuit(n, depth, seed=43)
    
    coupling = CouplingMap.from_line(n)
    print("Transpiling...")
    c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    c2 = transpile(c2, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    
    print("Running 20-qubit fidelity check...")
    torch.cuda.synchronize()
    
    print("Running with rSVD on cuda:1...")
    start_time = time.time()
    result = get_fidelity_gpu(c1, c2, use_randomized_svd=True, use_batched_svd=False, device='cuda:1')
    end_time = time.time()
    
    print(f"Fidelity: {result['fidelity']}")
    print(f"Time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    profile_20q()
