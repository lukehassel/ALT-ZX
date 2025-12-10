import os
import gc
import torch
import torch.nn as nn
import dgl
import numpy as np
import random
import pyzx as zx
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.equivalence_checker import MPO, iterate
from src.model.diffusion import DiscreteDiffusion
from mpo.circuit_utils import get_universal_gate_set
from mpo.circuit_utils import create_random_circuit_with_universal_gates
from mpo.fidelity import get_fidelity

GATE_TO_IDX = {g: i for i, g in enumerate(get_universal_gate_set()['all'])}
IDX_TO_GATE = {i: g for g, i in GATE_TO_IDX.items()}
NUM_GATE_TYPES = len(get_universal_gate_set()['all'])


class EncoderDataset(Dataset):
    def __init__(self, file_path=None, size=100, min_qubits=3, max_qubits=5, 
                 min_depth=5, max_depth=15, verbose=False,
                 chunk_size=1000):
        """
        Args:
            file_path (str): Path to save/load the dataset (e.g. 'data/train_v1.pt')
            size (int): Number of pairs to generate if file doesn't exist.
        """
        self.size = size
        self.verbose = verbose
        self.file_path = file_path
        self.stream_to_dir = (
            self.file_path is not None
            and not self.file_path.endswith(".pt")
        )
        self.chunk_size = chunk_size
        self._current_chunk = []
        self._chunk_idx = 0
        self.data_pairs = []
        self._chunks = []
        self._total_len = 0
        self.diffusion = None
        
        self._init_diffusion()
        self._generate_dataset(min_qubits, max_qubits, min_depth, max_depth)

        if self.file_path and not self.stream_to_dir:
            self._save_dataset()

    def _init_diffusion(self):
        marginal = torch.ones(NUM_GATE_TYPES) / NUM_GATE_TYPES
        self.diffusion = DiscreteDiffusion(marginal_list=[marginal], T=100)

    def _load_dataset(self):
        if self.verbose:
            print(f"Loading dataset from {self.file_path}...")
        self.data_pairs = torch.load(self.file_path)
        if self.verbose:
            print(f"Successfully loaded {len(self.data_pairs)} pairs.")

    def _save_dataset(self):
        """Save the entire in‑memory dataset to a single .pt file."""
        if self.verbose:
            print(f"Saving dataset to {self.file_path}...")

        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.data_pairs, self.file_path)
        if self.verbose:
            print("Dataset saved.")

    def _flush_chunk_to_disk(self):
        """Write the current in‑memory chunk to disk and clear it."""
        if not self._current_chunk:
            return

        assert self.stream_to_dir, "Chunk flushing only valid in streaming mode."

        os.makedirs(self.file_path, exist_ok=True)
        chunk_path = os.path.join(
            self.file_path,
            f"dataset_chunk_{self._chunk_idx:06d}.pt",
        )
        if self.verbose:
            print(f"Flushing {len(self._current_chunk)} samples to {chunk_path}...")

        torch.save(self._current_chunk, chunk_path)

        num = len(self._current_chunk)
        self._chunks.append((chunk_path, num))
        self._total_len += num

        self._current_chunk.clear()
        gc.collect()
        self._chunk_idx += 1
            
    def _apply_noise_to_circuit(self, qc, t_val=None, debug=False):
        dag = circuit_to_dag(qc)
        ops = list(dag.topological_op_nodes())
        if not ops: return qc
        
        qubit_map = {q: i for i, q in enumerate(qc.qubits)}
        gate_indices = [GATE_TO_IDX.get(op.name, 0) for op in ops]
        x_0 = torch.tensor(gate_indices).long().unsqueeze(1)
        

        t_val = random.randint(10, 30)
        t = torch.tensor([t_val])
        intensity = t.item() / self.diffusion.T
        MAX_DRIFT = 0.1
        drift_probability = intensity
        _, x_t = self.diffusion.apply_noise(x_0, t)
        noisy_indices = x_t.reshape(-1).tolist()
        
        noisy_qc = QuantumCircuit(qc.num_qubits)
        univ_sets = get_universal_gate_set()
        singles = set(univ_sets['single_qubit'])
        param_gates = set(univ_sets['parametric'])
        changes = 0
        
        for i, op in enumerate(ops):
            orig_name = op.name
            new_idx = noisy_indices[i]
            new_name = IDX_TO_GATE[new_idx]
            
            if (orig_name in singles) != (new_name in singles):
                final_name = orig_name 
            else:
                final_name = new_name 

            if final_name != orig_name: changes += 1
            
            current_params = []
            
            if final_name in param_gates:
                num_required = 1
                
                if orig_name in param_gates and hasattr(op.op, 'params') and len(op.op.params) > 0:
                    old_vals = op.op.params
                    new_vals = []
                    
                    if random.random() < drift_probability:
                        sigma = MAX_DRIFT * intensity
                        for k in range(num_required):
                            base = float(old_vals[k]) if k < len(old_vals) else 0.0
                            perturbation = np.random.normal(0, sigma)
                            val = (base + perturbation) % (2 * np.pi)
                            new_vals.append(val)
                    else:
                        new_vals = old_vals
                        
                    current_params = new_vals
                else:
                    current_params = np.random.uniform(0, 2*np.pi, num_required).tolist()

            qubits = [qubit_map[q] for q in op.qargs]

            try:
                if final_name == orig_name and (not current_params or current_params == op.op.params):
                    noisy_qc.append(op.op, op.qargs, op.cargs)
                else:
                    getattr(noisy_qc, final_name)(*current_params, *qubits)
            except Exception as e:
                if debug: print(f"Fallback on {final_name}: {e}")
                noisy_qc.append(op.op, op.qargs, op.cargs)

        if debug:
            print(f"Noise Step t={t.item()} | Swaps: {changes}/{len(ops)}")

        return noisy_qc

    def _generate_dataset(self, min_q, max_q, min_d, max_d):
        if self.verbose:
            print(f"Generating {self.size} pairs (ZX + Noise)...")

        pbar = tqdm(total=self.size, disable=not self.verbose)
        attempts = 0

        num_generated = 0

        while num_generated < self.size:
            attempts += 1
            if attempts > self.size * 20:
                print("\nTimeout: Could not generate enough valid pairs.")
                break
            
            # 1. Generate Random Base
            num_qubits = random.randint(min_q, max_q)
            depth = random.randint(min_d, max_d)
            circuit = create_random_circuit_with_universal_gates(num_qubits, depth)
            noisy_circuit = self._apply_noise_to_circuit(circuit)
            fid_res = get_fidelity(circuit, noisy_circuit)
            fidelity_value = fid_res['fidelity']
            sample = {
                "circuit_1": qasm2.dumps(circuit),
                "circuit_2": qasm2.dumps(noisy_circuit),
                "fidelity": fidelity_value,
            }

            if self.stream_to_dir:
                self._current_chunk.append(sample)
                if len(self._current_chunk) >= self.chunk_size:
                    self._flush_chunk_to_disk()
            else:
                self.data_pairs.append(sample)

            del circuit, noisy_circuit, fid_res, sample
            gc.collect()

            num_generated += 1
            pbar.update(1)

        if self.stream_to_dir:
            self._flush_chunk_to_disk()

        pbar.close()

    def __len__(self):
        if self.stream_to_dir:
            return self._total_len
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if not self.stream_to_dir:
            return self.data_pairs[idx]

        if idx < 0:
            idx = self._total_len + idx
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)

        offset = idx
        for chunk_path, length in self._chunks:
            if offset < length:
                chunk = torch.load(chunk_path)
                return chunk[offset]
            offset -= length

        raise IndexError(idx)


def collate_dict_batch(batch):
    """
    Custom collate function that returns dict directly for batch_size=1,
    or properly batches for larger batch sizes.
    """
    if len(batch) == 1:
        return batch[0]
    else:
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)


def collate_quantum_graphs(batch):
    """
    Collate function for EncoderDataset that converts QASM strings to DGL graphs.
    Returns ((g1_batch, t1_batch, l1_batch), (g2_batch, t2_batch, l2_batch))
    """
    from encoder.utils import qasm_to_dgl
    
    graphs_1, types_1, locs_1 = [], [], []
    graphs_2, types_2, locs_2 = [], [], []
    
    for item in batch:
        g1, t1, l1 = qasm_to_dgl(item['circuit_1'])
        graphs_1.append(g1)
        types_1.append(t1)
        locs_1.append(l1)
        
        g2, t2, l2 = qasm_to_dgl(item['circuit_2'])
        graphs_2.append(g2)
        types_2.append(t2)
        locs_2.append(l2)
    
    batched_g1 = dgl.batch(graphs_1)
    batched_g2 = dgl.batch(graphs_2)
    
    batched_t1 = torch.cat(types_1)
    batched_l1 = torch.cat(locs_1)
    batched_t2 = torch.cat(types_2)
    batched_l2 = torch.cat(locs_2)
    
    return (batched_g1, batched_t1, batched_l1), (batched_g2, batched_t2, batched_l2)


if __name__ == "__main__":
    DATA_DIR = "/Volumes/Samsung_T5/layerdag_dataset"

    dataset = EncoderDataset(
        file_path=DATA_DIR,
        size=9999999,
        verbose=True,
        chunk_size=1000,
    )
    
    print(f"\nDataset Ready. Total Size: {len(dataset)}")