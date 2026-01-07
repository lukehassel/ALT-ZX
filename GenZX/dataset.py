import os
import sys
import random
import torch
import multiprocessing as mp
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyzx as zx
from zx_loader import pyzx_graph_to_pyg

DEFAULT_ARGS = {
    "size": 50000,
    "save_path": "GenZX/genzx_dataset.pt",
    "workers": max(1, (os.cpu_count() or 2) - 1),
    "max_nodes": 64,
}


def create_random_circuit_graph(num_qubits, depth, seed=None):
    if seed is not None:
        random.seed(seed)
    
    circuit = zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=num_qubits,
        depth=depth,
        p_had=0.2,
        p_t=0.2
    )
    
    graph = circuit.to_graph()
    
    return graph


def check_gflow(graph):
    try:
        return zx.gflow.gflow(graph) is not None
    except Exception:
        return False


def init_worker(seed_offset):
    random.seed(seed_offset + os.getpid())
    torch.manual_seed(seed_offset + os.getpid())


def generate_small_graph(max_nodes):
    max_qubits = min(8, max(3, max_nodes // 10))
    max_depth = min(8, max(3, max_nodes // 10))
    
    num_qubits = random.randint(3, max_qubits)
    depth = random.randint(3, max_depth)
    seed = random.randint(0, 1_000_000)
    
    try:
        graph = create_random_circuit_graph(num_qubits, depth, seed=seed)
        
        if graph is None:
            return None
        
        if graph.num_vertices() >= max_nodes:
            return None
        
        if not check_gflow(graph):
            return None
        
        pyg_data = pyzx_graph_to_pyg(graph)
        
        if pyg_data is None:
            return None
        
        if pyg_data.x.shape[0] >= max_nodes:
            return None
        
        return pyg_data
        
    except Exception as e:
        return None


def worker_task(args):
    max_nodes = args
    return generate_small_graph(max_nodes)


def main(size, save_path, num_workers, max_nodes, chunk_size=5000):
    import gc
    
    print(f"Generating {size} ZX graphs with < {max_nodes} nodes...")
    print(f"Using {num_workers} workers, saving every {chunk_size} samples")
    
    dir_name = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    else:
        dir_name = "."
    
    data_list = []
    chunk_idx = 0
    total_saved = 0
    
    total_attempts = int(size * 3)
    
    with mp.Pool(num_workers, initializer=init_worker, initargs=(random.randint(0, 10000),)) as pool:
        pbar = tqdm(total=size, desc="Generating graphs")
        
        iterator = pool.imap_unordered(worker_task, [max_nodes] * total_attempts)
        
        for result in iterator:
            if result is not None:
                data_list.append(result)
                pbar.update(1)
                
                if len(data_list) >= chunk_size:
                    chunk_path = os.path.join(dir_name, f"{base_name}_chunk_{chunk_idx}.pt")
                    torch.save(data_list, chunk_path)
                    print(f"\nSaved chunk {chunk_idx} ({len(data_list)} samples) to {chunk_path}")
                    total_saved += len(data_list)
                    
                    data_list = []
                    gc.collect()
                    chunk_idx += 1
                
            if total_saved + len(data_list) >= size:
                break
        pbar.close()
    
    if data_list:
        if chunk_idx == 0:
            torch.save(data_list, save_path)
            print(f"Saved {len(data_list)} samples to {save_path}")
        else:
            chunk_path = os.path.join(dir_name, f"{base_name}_chunk_{chunk_idx}.pt")
            torch.save(data_list, chunk_path)
            print(f"Saved final chunk {chunk_idx} ({len(data_list)} samples) to {chunk_path}")
        total_saved += len(data_list)
    
    print(f"\nTotal saved: {total_saved} graphs in {max(1, chunk_idx + (1 if data_list else 0))} file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ZX graphs for GenZX training")
    parser.add_argument("--size", type=int, default=DEFAULT_ARGS["size"],
                        help="Number of graphs to generate")
    parser.add_argument("--save-path", type=str, default=DEFAULT_ARGS["save_path"],
                        help="Output file path")
    parser.add_argument("--workers", type=int, default=DEFAULT_ARGS["workers"],
                        help="Number of worker processes")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_ARGS["max_nodes"],
                        help="Maximum number of nodes per graph")
    args = parser.parse_args()
    
    main(size=args.size, save_path=args.save_path, 
         num_workers=args.workers, max_nodes=args.max_nodes)
