import os
import gc
import time
import sys
import multiprocessing as mp
import warnings
import uuid

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import numpy as np
import random
import pyzx as zx
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit import transpile, qasm2

warnings.filterwarnings('ignore', category=UserWarning, module='stevedore')
import logging
logging.getLogger('stevedore').setLevel(logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from circuit_utils import get_universal_gate_set
from circuit_utils import create_random_circuit_with_universal_gates
from zx_loader import circuit_to_pyg

PYZX_BASIS_GATES = ['cx', 'cz', 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'ccx', 'swap']


def _init_worker(seed_offset):
    worker_seed = seed_offset + os.getpid()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
    print(f"Worker {os.getpid()} ready (CPU-only)", flush=True)


def _optimize_with_qiskit(circuit, optimization_level=3):
    return transpile(
        circuit, 
        basis_gates=PYZX_BASIS_GATES, 
        optimization_level=optimization_level
    )


def _optimize_with_pyzx(circuit):
    try:
        qasm_str = qasm2.dumps(circuit)
        pyzx_circuit = zx.Circuit.from_qasm(qasm_str)
        graph = pyzx_circuit.to_graph()
        zx.full_reduce(graph)
        optimized_pyzx = zx.Circuit.from_graph(graph)
        optimized_qasm = optimized_pyzx.to_qasm()
        return qasm2.loads(optimized_qasm)
    except Exception:
        return circuit


def _generate_single_optimized_sample(args):
    min_q, max_q, min_d, max_d, max_attempts, optimization_mode = args
    pid = os.getpid()
    
    for attempt in range(max_attempts):
        try:
            num_qubits = random.randint(min_q, max_q)
            depth = random.randint(min_d, max_d)
            
            original_circuit = create_random_circuit_with_universal_gates(num_qubits, depth)
            
            if optimization_mode == 'both':
                optimized_circuit = _optimize_with_qiskit(original_circuit, optimization_level=3)
                optimized_circuit = _optimize_with_pyzx(optimized_circuit)
            elif optimization_mode == 'qiskit':
                optimized_circuit = _optimize_with_qiskit(original_circuit, optimization_level=3)
            elif optimization_mode == 'pyzx':
                optimized_circuit = _optimize_with_pyzx(original_circuit)
            else:
                raise ValueError(f"Unknown optimization mode: {optimization_mode}")
            
            c1_trans = transpile(original_circuit, basis_gates=PYZX_BASIS_GATES, optimization_level=0)
            c2_trans = transpile(optimized_circuit, basis_gates=PYZX_BASIS_GATES, optimization_level=0)
            
            data1 = circuit_to_pyg(c1_trans)
            data2 = circuit_to_pyg(c2_trans)
            data1.num_qubits = c1_trans.num_qubits
            data2.num_qubits = c2_trans.num_qubits
            
            # Fidelity = 1.0 since optimizations are equivalence-preserving
            return (data1, data2, torch.tensor(1.0, dtype=torch.float32))
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Worker {pid} failed after {max_attempts} attempts: {e}", flush=True)
            continue
    return None


class OptimizationDataset(Dataset):
    def __init__(self, file_path=None, size=100, min_qubits=20, max_qubits=30,
                 min_depth=5, max_depth=10, verbose=False, optimization_mode='both',
                 chunk_size=1000, num_workers=None, timeout_minutes=30):
        self.size = size
        self.verbose = verbose
        self.file_path = file_path
        self.stream_to_dir = self.file_path is not None and not self.file_path.endswith(".pt")
        self.chunk_size = chunk_size
        self._current_chunk, self._chunk_idx, self.data_pairs, self._chunks, self._total_len = [], 0, [], [], 0
        self.optimization_mode = optimization_mode
        self.timeout_minutes = timeout_minutes
        self._start_time = time.time()
        
        if num_workers is None:
            try: num_gpus = torch.cuda.device_count()
            except: num_gpus = 0
            if num_gpus > 0: self.num_workers = 4 * num_gpus
            else:
                s_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
                self.num_workers = int(s_cpus) - 1 if s_cpus else mp.cpu_count() - 1
        else: self.num_workers = num_workers
        self.num_workers = max(1, self.num_workers)

        self._generate_dataset_parallel(min_qubits, max_qubits, min_depth, max_depth)
        if self.file_path and not self.stream_to_dir: self._save_dataset()

    def _save_dataset(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        torch.save(self.data_pairs, self.file_path)

    def _flush_chunk_to_disk(self):
        if not self._current_chunk: return
        os.makedirs(self.file_path, exist_ok=True)
        path = os.path.join(self.file_path, f"optim_chunk_{uuid.uuid4().hex[:8]}_{self._chunk_idx:06d}.pt")
        torch.save(self._current_chunk, path)
        print(f"Flushed chunk {self._chunk_idx} ({len(self._current_chunk)} items) to {path}", flush=True)
        self._chunks.append((path, len(self._current_chunk)))
        self._total_len += len(self._current_chunk)
        self._current_chunk.clear()
        self._chunk_idx += 1
        gc.collect()

    def _generate_dataset_parallel(self, min_q, max_q, min_d, max_d):
        if self.verbose: 
            print(f"Generating {self.size} optimization samples using {self.num_workers} workers "
                  f"(timeout: {self.timeout_minutes} min, mode: {self.optimization_mode})...", flush=True)
        args = (min_q, max_q, min_d, max_d, 5, self.optimization_mode)
        ctx = mp.get_context('spawn')
        pbar = tqdm(total=self.size, disable=not self.verbose)
        timeout_reached = False
        timeout_seconds = self.timeout_minutes * 60
        
        with ctx.Pool(processes=self.num_workers, initializer=_init_worker, initargs=(random.randint(0, 2**31),)) as pool:
            for result in pool.imap_unordered(_generate_single_optimized_sample, [args] * self.size):
                elapsed = time.time() - self._start_time
                if elapsed >= timeout_seconds:
                    print(f"\n[TIMEOUT] {self.timeout_minutes} minutes elapsed. Saving data and exiting...", flush=True)
                    timeout_reached = True
                    break
                    
                if result:
                    if self.stream_to_dir:
                        self._current_chunk.append(result)
                        if len(self._current_chunk) >= self.chunk_size: self._flush_chunk_to_disk()
                    else: self.data_pairs.append(result)
                    pbar.update(1)
                    if len(self.data_pairs) + self._total_len >= self.size: break
                    
        if self.stream_to_dir: self._flush_chunk_to_disk()
        pbar.close()
        
        elapsed_total = time.time() - self._start_time
        if timeout_reached:
            print(f"[TIMEOUT] Saved {self._total_len} optimization samples in {elapsed_total/60:.1f} minutes.", flush=True)
        else:
            print(f"Completed {self._total_len} optimization samples in {elapsed_total/60:.1f} minutes.", flush=True)

    def __len__(self): return self._total_len if self.stream_to_dir else len(self.data_pairs)
    
    def __getitem__(self, idx):
        if not self.stream_to_dir: return self.data_pairs[idx]
        for path, length in self._chunks:
            if idx < length: return torch.load(path)[idx]
            idx -= length
        raise IndexError(idx)


if __name__ == "__main__":
    print("Warmup: Importing dependencies...", flush=True)
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=28)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "qiskit", "pyzx"])
    parser.add_argument("--output-dir", type=str, default="./optimization_data")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--restart-count", type=int, default=0)
    args = parser.parse_args()
    
    print(f"[Restart #{args.restart_count}] Starting optimization dataset generation "
          f"(timeout: {args.timeout} min, mode: {args.mode})...", flush=True)
    
    dataset = OptimizationDataset(
        file_path=args.output_dir, 
        size=args.size, 
        num_workers=args.workers, 
        verbose=args.verbose,
        optimization_mode=args.mode,
        timeout_minutes=args.timeout
    )
    print(f"[Restart #{args.restart_count}] Saved {len(dataset)} optimization samples.", flush=True)
    
    if args.restart_count < args.max_restarts:
        print(f"\n[AUTO-RESTART] Spawning new process (restart #{args.restart_count + 1})...", flush=True)
        time.sleep(2)
        
        cmd = [
            sys.executable, "-u", __file__,
            "--size", str(args.size),
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--mode", args.mode,
            "--output-dir", args.output_dir,
            "--max-restarts", str(args.max_restarts),
            "--restart-count", str(args.restart_count + 1),
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        os.execv(sys.executable, cmd)
    else:
        print(f"[COMPLETE] Reached max restarts ({args.max_restarts}). Exiting.", flush=True)
