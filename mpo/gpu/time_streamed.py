import torch
import time
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from mpo.gpu.mpo_gpu import get_fidelity_gpu


def run_timing():
    n = 14 
    depth = 5
    basis_gates = ['cx', 'h', 'rz', 'sx', 'x']
    
    print(f"Generating random circuit (n={n}, d={depth})...")
    c1 = random_circuit(n, depth, seed=42)
    c2 = random_circuit(n, depth, seed=43)
    
    coupling = CouplingMap.from_line(n)
    c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    c2 = transpile(c2, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    
    print("Warming up...")
    try:
        c_warm = random_circuit(4, 2)
        c_warm = transpile(c_warm, coupling_map=CouplingMap.from_line(4), basis_gates=basis_gates, optimization_level=0)
        get_fidelity_gpu(c_warm, c_warm, transpile_to_linear=False)
    except Exception as e:
        print(f"Warmup failed: {e}")

    print("Running timed execution...")
    torch.cuda.synchronize()
    t0 = time.time()
    
    res = get_fidelity_gpu(c1, c2, transpile_to_linear=False)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"Time: {t1 - t0:.4f}s")
    print(f"Fidelity: {res['fidelity']}")


if __name__ == "__main__":
    run_timing()
