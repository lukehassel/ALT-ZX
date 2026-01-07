import torch
import numpy as np
import time
import warnings
from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import sys
import os

sys.path.append(os.getcwd())

from mpo.gpu.mpo_gpu import get_fidelity_gpu, MPOTensorGPU, iterate_gpu


def benchmark_single_gpu(n_qubits=20, depth=10, use_batched_svd=True):
    device = 'cuda:0'
    print(f"=== Single-GPU Benchmark: {n_qubits} qubits, depth {depth} ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batched SVD: {use_batched_svd}")
    
    basis = ['cx', 'h', 'rz', 'sx', 'x']
    coupling = CouplingMap.from_line(n_qubits)
    
    print("Generating random circuits...")
    c1 = random_circuit(n_qubits, depth, seed=42)
    c2 = random_circuit(n_qubits, depth, seed=43)
    
    print("Transpiling...")
    c1 = transpile(c1, coupling_map=coupling, basis_gates=basis, optimization_level=0)
    c2 = transpile(c2, coupling_map=coupling, basis_gates=basis, optimization_level=0)
    
    print(f"Circuit 1 depth: {c1.depth()}")
    
    print("Warmup (10 qubits)...")
    c1_small = random_circuit(10, 2, seed=42)
    c2_small = random_circuit(10, 2, seed=43)
    c1_small = transpile(c1_small, coupling_map=CouplingMap.from_line(10), basis_gates=basis)
    c2_small = transpile(c2_small, coupling_map=CouplingMap.from_line(10), basis_gates=basis)
    get_fidelity_gpu(c1_small, c2_small, device=device, use_batched_svd=use_batched_svd)
    
    print("Running main benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    result = get_fidelity_gpu(c1, c2, device=device, use_batched_svd=use_batched_svd)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"Fidelity: {result['fidelity']:.8f}")
    print(f"Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    benchmark_single_gpu(n_qubits=20, depth=5, use_batched_svd=True)
