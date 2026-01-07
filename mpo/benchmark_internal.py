import time
import torch
import numpy as np
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from mpo.gpu.mpo_gpu import get_fidelity_gpu, DTYPE, DEVICE


def benchmark():
    print(f"Current DTYPE: {DTYPE}")
    print(f"Device: {DEVICE}")
    
    n_qubits = 15
    depth = 20
    
    c1 = random_circuit(n_qubits, depth, seed=42)
    c2 = random_circuit(n_qubits, depth, seed=43)
    
    print("Transpiling...")
    c1 = transpile(c1, basis_gates=['cx', 'rx', 'ry', 'rz'], optimization_level=0)
    c2 = transpile(c2, basis_gates=['cx', 'rx', 'ry', 'rz'], optimization_level=0)
    
    print("Warming up...")
    try:
        get_fidelity_gpu(c1, c2)
    except Exception as e:
        print(f"Warmup failed: {e}")

    print("Benchmarking...")
    t0 = time.time()
    n_runs = 5
    for i in range(n_runs):
        get_fidelity_gpu(c1, c2)
        print(f"Run {i+1} done")
    
    elapsed = time.time() - t0
    avg_time = elapsed / n_runs
    print(f"Average time per run: {avg_time:.4f}s")


if __name__ == "__main__":
    benchmark()
