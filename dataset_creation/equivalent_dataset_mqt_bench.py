import torch
import pyzx as zx
from mqt.bench import get_benchmark, BenchmarkLevel
from qiskit import qasm2
from torch_geometric.data import Data
from pyzx.utils import VertexType, EdgeType
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zxnet_features import extract_zxnet_features

MIN_QUBITS = 4
MAX_QUBITS = 20
OUTPUT_DIR = "zxnet_dataset"
OPT_LEVELS = [0, 1, 2, 3]

BENCHMARKS = [
    "ae", "bv", "dj", "ghz", "graphstate",
    "qft", "qftentangled", "qpeexact", "qpeinexact", 
    "qaoa", "wstate", "vqe_real_amp", "vqe_su2", "vqe_two_local"
]


def generate_processed_dataset():
    label = torch.tensor(1.0, dtype=torch.float32)
    total_pairs = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating dataset: {len(BENCHMARKS)} benchmarks × {MAX_QUBITS - MIN_QUBITS + 1} sizes × opt_levels {OPT_LEVELS}", flush=True)
    print(f"Saving to: {OUTPUT_DIR}/", flush=True)
    
    for name in BENCHMARKS:
        for n_qubits in range(MIN_QUBITS, MAX_QUBITS + 1):
            output_path = os.path.join(OUTPUT_DIR, f"{name}_{n_qubits}q.pt")
            
            if os.path.exists(output_path):
                print(f"{name} ({n_qubits}q): already exists, skipping", flush=True)
                continue
            
            try:
                circuits = [
                    get_benchmark(name, level=BenchmarkLevel.INDEP, circuit_size=n_qubits, opt_level=opt)
                    for opt in OPT_LEVELS
                ]
                
                data_list = []
                for qc in circuits:
                    try:
                        g = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_graph()
                        data = extract_zxnet_features(g)
                        data_list.append(data)
                    except Exception as e:
                        print(f"    Failed to convert circuit: {e}")
                        continue
                
                if len(data_list) != len(circuits):
                    print(f"    Skipping {name} ({n_qubits}q) due to conversion failures", flush=True)
                    continue
                
                pairs = []
                for i in range(len(data_list)):
                    for j in range(i + 1, len(data_list)):
                        pairs.append((data_list[i], data_list[j], label))
                
                torch.save(pairs, output_path)
                total_pairs += len(pairs)
                
                print(f"{name} ({n_qubits}q): {len(pairs)} pairs saved (total: {total_pairs})", flush=True)
                
                del circuits, data_list, pairs
                
            except Exception as e:
                print(f"  Skip {name} ({n_qubits}q): {e}", flush=True)

    print(f"Done. Total: {total_pairs} pairs in {OUTPUT_DIR}/", flush=True)


def combine_dataset():
    all_pairs = []
    pt_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt') and f != 'combined.pt'])
    
    print(f"Combining {len(pt_files)} files from {OUTPUT_DIR}/", flush=True)
    
    for filename in pt_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        pairs = torch.load(filepath, weights_only=False)
        all_pairs.extend(pairs)
        print(f"  Loaded {filename}: {len(pairs)} pairs", flush=True)
    
    output_path = os.path.join(OUTPUT_DIR, "combined.pt")
    torch.save(all_pairs, output_path)
    print(f"Done. Combined {len(all_pairs)} pairs into {output_path}", flush=True)
    
    return all_pairs


if __name__ == "__main__":
    generate_processed_dataset()
    combine_dataset()