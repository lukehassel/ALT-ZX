import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from qiskit.circuit.random import random_circuit
from qiskit import transpile
from .mpo_gpu import get_fidelity_gpu


def diagnose_cuda():
    device = 'cuda:1'
    
    print("=== CUDA Diagnostic ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    print()
    
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    print(f"GPU memory before: {mem_before / 1024**2:.1f} MB")
    
    print("\nGenerating 20-qubit circuit...")
    c1 = random_circuit(20, 20, max_operands=2, seed=42)
    c2 = c1.copy()
    basis = ['cx', 'h', 'rz', 'sx', 'x', 'id']
    c1 = transpile(c1, basis_gates=basis, optimization_level=0)
    c2 = transpile(c2, basis_gates=basis, optimization_level=0)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print("\nRunning fidelity calculation...")
    
    torch.cuda.synchronize(device)
    wall_start = time.time()
    start_event.record()
    
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = get_fidelity_gpu(c1, c2, device=device, use_randomized_svd=True)
    
    end_event.record()
    torch.cuda.synchronize(device)
    wall_end = time.time()
    
    gpu_time_ms = start_event.elapsed_time(end_event)
    wall_time_s = wall_end - wall_start
    
    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)
    
    print("\n=== Results ===")
    print(f"Fidelity: {result['fidelity']:.6f}")
    print()
    print("=== Timing ===")
    print(f"Wall clock time: {wall_time_s:.3f} s")
    print(f"GPU kernel time: {gpu_time_ms:.1f} ms ({gpu_time_ms/1000:.3f} s)")
    print(f"CPU overhead:    {(wall_time_s - gpu_time_ms/1000)*1000:.1f} ms ({wall_time_s - gpu_time_ms/1000:.3f} s)")
    print(f"GPU efficiency:  {(gpu_time_ms/1000) / wall_time_s * 100:.1f}%")
    print()
    print("=== GPU Memory ===")
    print(f"Memory before: {mem_before / 1024**2:.1f} MB")
    print(f"Memory after:  {mem_after / 1024**2:.1f} MB")
    print(f"Peak memory:   {mem_peak / 1024**2:.1f} MB")
    print(f"Memory used:   {(mem_peak - mem_before) / 1024**2:.1f} MB")
    print()
    
    if gpu_time_ms/1000 < wall_time_s * 0.1:
        print("⚠️  WARNING: Less than 10% of time is spent on GPU!")
        print("    The workload is CPU-bound (Python orchestration overhead).")
    elif gpu_time_ms/1000 < wall_time_s * 0.5:
        print("⚠️  WARNING: Less than 50% of time is spent on GPU!")
        print("    Significant CPU overhead exists.")
    else:
        print("✓  Good GPU utilization - most time is spent on GPU kernels.")


if __name__ == "__main__":
    diagnose_cuda()
