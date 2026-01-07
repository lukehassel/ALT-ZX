import torch
import time
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from mpo.gpu.mpo_gpu import get_fidelity_gpu


def run_profile():
    print("Generating circuit for batched profiling...")
    n = 12
    depth = 5
    basis_gates = ['cx', 'h', 'rz', 'sx', 'x']
    
    c1 = random_circuit(n, depth, seed=42)
    c2 = random_circuit(n, depth, seed=43)
    
    coupling = CouplingMap.from_line(n)
    c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    c2 = transpile(c2, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
    
    print("Starting profiling...")
    try:
        get_fidelity_gpu(c1, c2, transpile_to_linear=False)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        get_fidelity_gpu(c1, c2, transpile_to_linear=False)
        
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    run_profile()
