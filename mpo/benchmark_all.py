import time
import torch
import numpy as np
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap

from mpo.fidelity import get_fidelity as get_fidelity_cpu
from mpo.fidelity_gpu import get_fidelity_fast as get_fidelity_sv_gpu
from mpo.gpu.mpo_gpu import get_fidelity_gpu as get_fidelity_mpo_gpu


def run_benchmark(num_qubits_list, depth=5):
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    print(f"Depth: {depth}\n")
    
    basis_gates = ['cx', 'h', 'rz', 'sx', 'x']
    
    for n in num_qubits_list:
        print(f"--- Qubits: {n} ---")
        c1 = random_circuit(n, depth, seed=42)
        c2 = random_circuit(n, depth, seed=43)
        
        coupling = CouplingMap.from_line(n)
        c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
        c2 = transpile(c2, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
        
        if n <= 8:
            print(f"[{n}] Running CPU MPO...", flush=True, end="")
            t0 = time.time()
            try:
                res_cpu = get_fidelity_cpu(c1, c2)
                t_cpu = time.time() - t0
                print(f" done ({t_cpu:.2f}s)", flush=True)
            except Exception as e:
                print(f" FAILED: {e}", flush=True)
                t_cpu = float('nan')
                res_cpu = {'fidelity': float('nan')}
        else:
            print(f"[{n}] Skipping CPU MPO (too slow)", flush=True)
            t_cpu = float('nan')
            res_cpu = {'fidelity': float('nan')}
        
        print(f"[{n}] Running SV GPU...", flush=True, end="")
        t0 = time.time()
        try:
            res_sv = get_fidelity_sv_gpu(c1, c2, method='statevector_gpu')
            t_sv = time.time() - t0
            print(f" done ({t_sv:.2f}s)", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            t_sv = float('nan')
        
        import mpo.mpo_gpu as mpo_gpu_mod
        old_cuda = mpo_gpu_mod.MPO_CUDA
        mpo_gpu_mod.MPO_CUDA = None
        
        print(f"[{n}] Running Fallback MPO...", flush=True, end="")
        t0 = time.time()
        try:
            res_fallback = get_fidelity_mpo_gpu(c1, c2, transpile_to_linear=False)
            t_fallback = time.time() - t0
            print(f" done ({t_fallback:.2f}s)", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            t_fallback = float('nan')
        
        mpo_gpu_mod.MPO_CUDA = old_cuda
        
        print(f"[{n}] Running CUDA MPO...", flush=True, end="")
        t0 = time.time()
        try:
            res_cuda = get_fidelity_mpo_gpu(c1, c2, transpile_to_linear=False)
            t_cuda = time.time() - t0
            print(f" done ({t_cuda:.2f}s)", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            t_cuda = float('nan')
            res_cuda = {'fidelity': float('nan')}
        
        results.append({
            'Qubits': n,
            'CPU': t_cpu,
            'SV': t_sv,
            'Fallback': t_fallback,
            'CUDA': t_cuda
        })
        
    return results


if __name__ == "__main__":
    qubits = [4, 8, 12, 16, 20, 24]
    results = run_benchmark(qubits, depth=5)
    
    if results:
        print(f"\nFinal Summary (seconds):")
        print(f"{'Qubits':<8} {'CPU MPO':<12} {'SV GPU':<12} {'Fallback':<12} {'CUDA MPO':<12}")
        for r in results:
             print(f"{r['Qubits']:<8} {r['CPU']:<12.4f} {r['SV']:<12.4f} {r['Fallback']:<12.4f} {r['CUDA']:<12.4f}")

        valid_fallback = [r for r in results if not np.isnan(r['Fallback']) and not np.isnan(r['CUDA'])]
        if valid_fallback:
            avg_fallback_speedup = sum(r['Fallback']/r['CUDA'] for r in valid_fallback) / len(valid_fallback)
            print(f"\nAvg Speedup vs Fallback MPO: {avg_fallback_speedup:.2f}x")
        
        valid_cpu = [r for r in results if not np.isnan(r['CPU']) and not np.isnan(r['CUDA'])]
        if valid_cpu:
            avg_cpu_speedup = sum(r['CPU']/r['CUDA'] for r in valid_cpu) / len(valid_cpu)
            print(f"Avg Speedup vs CPU MPO: {avg_cpu_speedup:.2f}x")
