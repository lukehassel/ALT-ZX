import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from qiskit.circuit.random import random_circuit
from qiskit import transpile
from .mpo_gpu import get_fidelity_gpu


def profile_with_nsight():
    device = 'cuda:3'
    
    print("Generating circuit...")
    n_qubits = 20
    depth = 20
    c1 = random_circuit(n_qubits, depth, max_operands=2, seed=42)
    c2 = c1.copy()
    
    basis = ['cx', 'h', 'rz', 'sx', 'x', 'id', 's', 'sdg', 't', 'tdg', 'u1', 'u2', 'u3', 'ry', 'rx']
    c1 = transpile(c1, basis_gates=basis, optimization_level=0)
    c2 = transpile(c2, basis_gates=basis, optimization_level=0)
    
    print(f"Circuit: {n_qubits} qubits, {c1.depth()} depth, {len(c1.data)} gates")
    
    print("Warmup run...")
    _ = get_fidelity_gpu(c1, c2, device=device, use_randomized_svd=True)
    torch.cuda.synchronize()
    
    print("Profiling with torch.profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = get_fidelity_gpu(c1, c2, device=device, use_randomized_svd=True)
        torch.cuda.synchronize()
    
    print(f"\nFidelity: {result['fidelity']:.6f}")
    
    print("\n=== Top 20 CUDA Operations by Self CUDA Time ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    
    print("\n=== Top 20 CPU Operations by Self CPU Time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    
    trace_path = "/home/wo057552/ALT-ZX/mpo/profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace exported to: {trace_path}")


if __name__ == "__main__":
    profile_with_nsight()
