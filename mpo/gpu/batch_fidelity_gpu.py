import torch
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import threading

sys.path.insert(0, os.path.dirname(__file__))

from qiskit.circuit.random import random_circuit
from qiskit import transpile
from .mpo_gpu import get_fidelity_gpu

_thread_local = threading.local()


def get_thread_stream(device: str) -> torch.cuda.Stream:
    if not hasattr(_thread_local, 'streams'):
        _thread_local.streams = {}
    if device not in _thread_local.streams:
        _thread_local.streams[device] = torch.cuda.Stream(device=device)
    return _thread_local.streams[device]


def compute_fidelity_in_stream(circuit1, circuit2, device: str, use_randomized_svd: bool = True) -> dict:
    stream = get_thread_stream(device)
    with torch.cuda.stream(stream):
        result = get_fidelity_gpu(circuit1, circuit2, device=device, use_randomized_svd=use_randomized_svd)
    stream.synchronize()
    return result


def generate_circuit_pair(n_qubits: int, depth: int, seed: int, same_circuit: bool = True) -> Tuple:
    from qiskit.transpiler import CouplingMap
    
    basis = ['cx', 'h', 'rz', 'sx', 'x']
    coupling = CouplingMap.from_line(n_qubits)
    
    c1 = random_circuit(n_qubits, depth, max_operands=2, seed=seed)
    if same_circuit:
        c2 = c1.copy()
    else:
        c2 = random_circuit(n_qubits, depth, max_operands=2, seed=seed + 1000)
    
    c1 = transpile(c1, coupling_map=coupling, basis_gates=basis, optimization_level=0)
    c2 = transpile(c2, coupling_map=coupling, basis_gates=basis, optimization_level=0)
    return c1, c2


def batch_fidelity_gpu(circuit_pairs: List[Tuple], device: str = 'cuda:3', 
                       num_workers: int = 8, use_randomized_svd: bool = True) -> List[dict]:
    results = [None] * len(circuit_pairs)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(compute_fidelity_in_stream, c1, c2, device, use_randomized_svd): idx
            for idx, (c1, c2) in enumerate(circuit_pairs)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {'fidelity': float('nan'), 'time': 0, 'error': str(e)}
    
    return results


def batch_fidelity_multi_gpu(circuit_pairs: List[Tuple], devices: List[str] = ['cuda:1', 'cuda:3'],
                             num_workers_per_gpu: int = 4, use_randomized_svd: bool = True) -> List[dict]:
    n_pairs = len(circuit_pairs)
    n_gpus = len(devices)
    results = [None] * n_pairs
    
    # Round-robin assignment for load balancing
    gpu_assignments = [[] for _ in range(n_gpus)]
    for idx, pair in enumerate(circuit_pairs):
        gpu_idx = idx % n_gpus
        gpu_assignments[gpu_idx].append((idx, pair))
    
    def process_on_device_serial(device: str, assignments: List[Tuple]) -> List[Tuple[int, dict]]:
        device_results = []
        for idx, pair in assignments:
            try:
                result = get_fidelity_gpu(pair[0], pair[1], device=device, use_randomized_svd=use_randomized_svd)
            except Exception as e:
                result = {'fidelity': float('nan'), 'time': 0, 'error': str(e)}
            device_results.append((idx, result))
        torch.cuda.synchronize(device)
        return device_results
    
    with ThreadPoolExecutor(max_workers=n_gpus) as gpu_executor:
        gpu_futures = {
            gpu_executor.submit(process_on_device_serial, devices[gpu_idx], gpu_assignments[gpu_idx]): gpu_idx
            for gpu_idx in range(n_gpus)
        }
        
        for future in as_completed(gpu_futures):
            device_results = future.result()
            for idx, result in device_results:
                results[idx] = result
    
    return results


def benchmark_gpu_saturation(n_circuits: int = 16, n_qubits: int = 20, depth: int = 20,
                             device: str = 'cuda:3', num_workers: int = 8):
    print(f"=== GPU Saturation Benchmark ===")
    print(f"Circuits: {n_circuits} pairs")
    print(f"Qubits: {n_qubits}, Depth: {depth}")
    print(f"Workers: {num_workers}")
    print(f"Device: {device}")
    print()
    
    print("Generating circuits...")
    t0 = time.time()
    circuit_pairs = [generate_circuit_pair(n_qubits, depth, seed=i) for i in range(n_circuits)]
    print(f"Generation time: {time.time() - t0:.2f}s")
    
    print("Warmup...")
    _ = get_fidelity_gpu(circuit_pairs[0][0], circuit_pairs[0][1], device=device)
    torch.cuda.synchronize()
    
    print("\n--- Sequential Execution ---")
    t0 = time.time()
    seq_results = []
    for c1, c2 in circuit_pairs:
        res = get_fidelity_gpu(c1, c2, device=device, use_randomized_svd=True)
        seq_results.append(res)
        torch.cuda.synchronize()
    seq_time = time.time() - t0
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Throughput: {n_circuits / seq_time:.2f} circuits/s")
    
    print(f"\n--- Concurrent Execution ({num_workers} workers) ---")
    t0 = time.time()
    par_results = batch_fidelity_gpu(circuit_pairs, device=device, num_workers=num_workers, use_randomized_svd=True)
    par_time = time.time() - t0
    print(f"Concurrent time: {par_time:.2f}s")
    print(f"Throughput: {n_circuits / par_time:.2f} circuits/s")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    print("\n--- Verification ---")
    all_match = True
    for i, (sr, pr) in enumerate(zip(seq_results, par_results)):
        if abs(sr['fidelity'] - pr['fidelity']) > 1e-6:
            print(f"Mismatch at {i}: seq={sr['fidelity']:.6f}, par={pr['fidelity']:.6f}")
            all_match = False
    
    if all_match:
        print("All results match!")
    
    return {
        'sequential_time': seq_time,
        'concurrent_time': par_time,
        'speedup': seq_time / par_time,
        'throughput': n_circuits / par_time
    }


def benchmark_multi_gpu(n_circuits: int = 16, n_qubits: int = 10, depth: int = 5,
                        devices: List[str] = ['cuda:1', 'cuda:3'], num_workers_per_gpu: int = 4,
                        use_randomized_svd: bool = True):
    print(f"=== Multi-GPU Benchmark ===")
    print(f"Circuits: {n_circuits} pairs")
    print(f"Qubits: {n_qubits}, Depth: {depth}")
    print(f"Devices: {devices}")
    print(f"Workers per GPU: {num_workers_per_gpu}")
    print()
    
    print("Generating circuits...")
    t0 = time.time()
    circuit_pairs = [generate_circuit_pair(n_qubits, depth, seed=i) for i in range(n_circuits)]
    print(f"Generation time: {time.time() - t0:.2f}s")
    
    print("Warmup on all GPUs...")
    for device in devices:
        _ = get_fidelity_gpu(circuit_pairs[0][0], circuit_pairs[0][1], device=device)
    for device in devices:
        torch.cuda.synchronize(device)
    
    print(f"\n--- Single GPU ({devices[0]}) ---")
    t0 = time.time()
    single_results = []
    for c1, c2 in circuit_pairs:
        result = get_fidelity_gpu(c1, c2, device=devices[0], use_randomized_svd=use_randomized_svd)
        single_results.append(result)
    torch.cuda.synchronize(devices[0])
    single_time = time.time() - t0
    print(f"Single GPU time: {single_time:.2f}s")
    print(f"Throughput: {n_circuits / single_time:.2f} circuits/s")
    
    print(f"\n--- Multi GPU ({', '.join(devices)}) ---")
    t0 = time.time()
    multi_results = batch_fidelity_multi_gpu(circuit_pairs, devices=devices,
                                              num_workers_per_gpu=num_workers_per_gpu, use_randomized_svd=True)
    multi_time = time.time() - t0
    print(f"Multi GPU time: {multi_time:.2f}s")
    print(f"Throughput: {n_circuits / multi_time:.2f} circuits/s")
    print(f"Speedup: {single_time / multi_time:.2f}x")
    
    print("\n--- Verification ---")
    all_match = True
    for i, (sr, mr) in enumerate(zip(single_results, multi_results)):
        diff = abs(sr['fidelity'] - mr['fidelity'])
        if diff > 1e-10:
            print(f"Mismatch at {i}: single={sr['fidelity']:.8f}, multi={mr['fidelity']:.8f}, diff={diff:.2e}")
            all_match = False
    
    if all_match:
        print("✓ All results match!")
    
    return {
        'single_gpu_time': single_time,
        'multi_gpu_time': multi_time,
        'speedup': single_time / multi_time,
        'throughput': n_circuits / multi_time
    }


if __name__ == "__main__":
    benchmark_multi_gpu(
        n_circuits=16,
        n_qubits=10,
        depth=5,
        devices=['cuda:1', 'cuda:3'],
        num_workers_per_gpu=4
    )
