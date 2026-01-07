import os
import gc
import time
import sys
import multiprocessing as mp
import warnings
import uuid

import torch
import torch.nn as nn
import numpy as np
import random
import pyzx as zx
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit import transpile, qasm2
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.equivalence_checker import MPO, iterate

warnings.filterwarnings('ignore', category=UserWarning, module='stevedore')
import logging
logging.getLogger('stevedore').setLevel(logging.ERROR)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model.diffusion import DiscreteDiffusion
from circuit_utils import get_universal_gate_set
from circuit_utils import create_random_circuit_with_universal_gates
from mpo.gpu.mpo_gpu import get_fidelity_gpu
from zx_loader import circuit_to_pyg
from encoder.noise import apply_noise_to_circuit, GATE_TO_IDX, IDX_TO_GATE, NUM_GATE_TYPES, _SINGLES, _PARAM_GATES

PYZX_BASIS_GATES = ['cx', 'cz', 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'ccx', 'swap']


def _init_worker(seed_offset):
    worker_seed = seed_offset + os.getpid()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
    
    time.sleep(random.uniform(0.1, 2.0))

    global WORKER_DEVICE
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            proc_name = mp.current_process().name
            try:
                rank = int(proc_name.split('-')[-1]) - 1
                device_id = rank % num_gpus
            except:
                device_id = os.getpid() % num_gpus
            WORKER_DEVICE = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(WORKER_DEVICE)
            torch.cuda.empty_cache()
        else:
            WORKER_DEVICE = torch.device('cpu')
    except Exception as e:
        print(f"Worker {os.getpid()} device init failed: {e}", flush=True)
        WORKER_DEVICE = torch.device('cpu')

    print(f"Worker {os.getpid()} ready on {WORKER_DEVICE}", flush=True)


def _generate_single_sample(args):
    min_q, max_q, min_d, max_d, max_attempts, use_pyzx = args
    pid = os.getpid()
    
    marginal = torch.ones(NUM_GATE_TYPES) / NUM_GATE_TYPES
    diffusion = DiscreteDiffusion(marginal_list=[marginal], T=100)
    
    for attempt in range(max_attempts):
        try:
            num_qubits = random.randint(min_q, max_q)
            depth = random.randint(min_d, max_d)
            circuit = create_random_circuit_with_universal_gates(num_qubits, depth)
            
            noisy_circuit = apply_noise_to_circuit(circuit, 10, diffusion)
            
            if use_pyzx:
                try:
                    clean_qasm = qasm2.dumps(circuit)
                    noisy_qasm = qasm2.dumps(noisy_circuit)
                    g_clean = zx.Circuit.from_qasm(clean_qasm).to_graph()
                    g_noisy = zx.Circuit.from_qasm(noisy_qasm).to_graph()
                    zx.full_reduce(g_clean)
                    zx.full_reduce(g_noisy)
                    circuit = qasm2.loads(zx.Circuit.from_graph(g_clean).to_qasm())
                    noisy_circuit = qasm2.loads(zx.Circuit.from_graph(g_noisy).to_qasm())
                except:
                    pass

            fid_res = get_fidelity_gpu(
                circuit, noisy_circuit, 
                device=WORKER_DEVICE, 
                use_randomized_svd=True,
                use_batched_svd=True
            )
            fidelity_value = fid_res["fidelity"]
            
            c1_trans = transpile(circuit, basis_gates=PYZX_BASIS_GATES, optimization_level=0)
            c2_trans = transpile(noisy_circuit, basis_gates=PYZX_BASIS_GATES, optimization_level=0)
            
            data1 = circuit_to_pyg(c1_trans)
            data2 = circuit_to_pyg(c2_trans)
            data1.num_qubits = c1_trans.num_qubits
            data2.num_qubits = c2_trans.num_qubits
            
            return (data1, data2, torch.tensor(fidelity_value, dtype=torch.float32))
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Worker {pid} failed after {max_attempts} attempts: {e}", flush=True)
            continue
    return None


class EncoderDataset(Dataset):
    def __init__(self, file_path=None, size=100, min_qubits=5, max_qubits=10,
                 min_depth=5, max_depth=10, verbose=False, use_pyzx=True,
                 chunk_size=200, num_workers=None, timeout_minutes=30):
        self.size = size
        self.verbose = verbose
        self.file_path = file_path
        self.stream_to_dir = self.file_path is not None and not self.file_path.endswith(".pt")
        self.chunk_size = chunk_size
        self._current_chunk, self._chunk_idx, self.data_pairs, self._chunks, self._total_len = [], 0, [], [], 0
        self.use_pyzx = use_pyzx
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
        path = os.path.join(self.file_path, f"chunk_{uuid.uuid4().hex[:8]}_{self._chunk_idx:06d}.pt")
        torch.save(self._current_chunk, path)
        print(f"Flushed chunk {self._chunk_idx} ({len(self._current_chunk)} items) to {path}", flush=True)
        self._chunks.append((path, len(self._current_chunk)))
        self._total_len += len(self._current_chunk)
        self._current_chunk.clear()
        self._chunk_idx += 1
        gc.collect()

    def _generate_dataset_parallel(self, min_q, max_q, min_d, max_d):
        if self.verbose: print(f"Generating {self.size} samples using {self.num_workers} workers (timeout: {self.timeout_minutes} min)...", flush=True)
        args = (min_q, max_q, min_d, max_d, 5, self.use_pyzx)
        ctx = mp.get_context('spawn')
        pbar = tqdm(total=self.size, disable=not self.verbose)
        timeout_reached = False
        timeout_seconds = self.timeout_minutes * 60
        
        with ctx.Pool(processes=self.num_workers, initializer=_init_worker, initargs=(random.randint(0, 2**31),)) as pool:
            for result in pool.imap_unordered(_generate_single_sample, [args] * self.size):
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
            print(f"[TIMEOUT] Saved {self._total_len} samples in {elapsed_total/60:.1f} minutes.", flush=True)
        else:
            print(f"Completed {self._total_len} samples in {elapsed_total/60:.1f} minutes.", flush=True)

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
    parser.add_argument("--pause", type=int, default=20)
    parser.add_argument("--do-pause", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--restart-count", type=int, default=0)
    args = parser.parse_args()
    
    if args.do_pause and args.pause > 0:
        print(f"\n[GPU COOLDOWN] Fresh process started. Pausing for {args.pause} minutes...", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        for minute in range(args.pause):
            time.sleep(60)
            remaining = args.pause - minute - 1
            if remaining > 0:
                print(f"[GPU COOLDOWN] {remaining} minutes remaining...", flush=True)
        print(f"[GPU COOLDOWN] Complete. Starting generation...", flush=True)
    
    print(f"[Restart #{args.restart_count}] Starting generation (timeout: {args.timeout} min)...", flush=True)
    
    dataset = EncoderDataset(
        file_path="./data", 
        size=args.size, 
        num_workers=args.workers, 
        verbose=args.verbose,
        timeout_minutes=args.timeout
    )
    print(f"[Restart #{args.restart_count}] Saved {len(dataset)} samples.", flush=True)
    
    if args.restart_count < args.max_restarts:
        print(f"\n[AUTO-RESTART] Spawning new process with cooldown (restart #{args.restart_count + 1})...", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)
        
        cmd = [
            sys.executable, "-u", __file__,
            "--size", str(args.size),
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--pause", str(args.pause),
            "--do-pause",
            "--max-restarts", str(args.max_restarts),
            "--restart-count", str(args.restart_count + 1),
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        os.execv(sys.executable, cmd)
    else:
        print(f"[COMPLETE] Reached max restarts ({args.max_restarts}). Exiting.", flush=True)