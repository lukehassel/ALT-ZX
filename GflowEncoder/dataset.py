import os
import sys
import random
import torch
import multiprocessing as mp
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zx_loader import pyzx_graph_to_pyg
from GflowNet.dataset import create_random_circuit_graph, apply_random_zx_transform, check_gflow

DEFAULT_ARGS = {
    "size": 50000,
    "save_path": "GflowEncoder/dataset.pt",
    "workers": os.cpu_count()-1 if os.cpu_count() else 1
}


def integration_init_worker(args):
    seed_offset = args
    random.seed(seed_offset + os.getpid())
    torch.manual_seed(seed_offset + os.getpid())


def integration_create_circuit(num_qubits, depth, seed):
    return create_random_circuit_graph(num_qubits, depth, seed=seed)


def integration_pyzx_to_pyg(graph):
    return pyzx_graph_to_pyg(graph)


def integration_apply_transform(graph):
    return apply_random_zx_transform(graph)


def integration_check_gflow(graph):
    return check_gflow(graph)


def integration_save_torch(data, path):
    torch.save(data, path)


def integration_makedirs(path):
    os.makedirs(path, exist_ok=True)


def integration_load_torch(path):
    return torch.load(path)


def generate_triplet(num_qubits_range, depth_range):
    num_qubits = random.randint(*num_qubits_range)
    depth = random.randint(*depth_range)
    seed = random.randint(0, 1_000_000)
    
    # Anchor: original valid circuit graph
    graph = integration_create_circuit(num_qubits, depth, seed)
    anchor_data = integration_pyzx_to_pyg(graph)
    if anchor_data is None:
        return None
    
    # Positive: 2 random ZX transforms (should preserve gflow)
    pos_graph = graph.copy()
    for _ in range(2):
        pos_graph = integration_apply_transform(pos_graph)
    
    if not integration_check_gflow(pos_graph):
        return None
        
    pos_data = integration_pyzx_to_pyg(pos_graph)
    if pos_data is None:
        return None
        
    # Negative: 5-50 random transforms (should break gflow)
    neg_graph = graph.copy()
    num_transforms = random.randint(5, 50)
    for _ in range(num_transforms):
        neg_graph = integration_apply_transform(neg_graph)
    
    if integration_check_gflow(neg_graph):
        return None
        
    neg_data = integration_pyzx_to_pyg(neg_graph)
    if neg_data is None:
        return None
        
    return (anchor_data, pos_data, neg_data)


def _worker_task(args):
    return generate_triplet(*args)


def main(size, save_path, num_workers, chunk_size=5000):
    import gc
    
    print(f"Generating {size} triplets using {num_workers} workers...")
    print(f"Saving every {chunk_size} samples...")
    
    num_qubits_range = (3, 15)
    depth_range = (5, 15)
    args = (num_qubits_range, depth_range)
    
    dir_name = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    if dir_name:
        integration_makedirs(dir_name)
    else:
        dir_name = "."
    
    data_list = []
    chunk_idx = 0
    total_saved = 0
    
    with mp.Pool(num_workers, initializer=integration_init_worker, initargs=(random.randint(0, 10000),)) as pool:
        pbar = tqdm(total=size)
        total_attempts = int(size * 1.5)
        
        iterator = pool.imap_unordered(_worker_task, [args] * total_attempts)
        
        for result in iterator:
            if result is not None:
                data_list.append(result)
                pbar.update(1)
                
                if len(data_list) >= chunk_size:
                    chunk_path = os.path.join(dir_name, f"{base_name}_chunk_{chunk_idx}.pt")
                    integration_save_torch(data_list, chunk_path)
                    print(f"\nSaved chunk {chunk_idx} ({len(data_list)} samples) to {chunk_path}")
                    total_saved += len(data_list)
                    
                    data_list = []
                    gc.collect()
                    chunk_idx += 1
                
            if total_saved + len(data_list) >= size:
                break
        pbar.close()
    
    if data_list:
        chunk_path = os.path.join(dir_name, f"{base_name}_chunk_{chunk_idx}.pt")
        integration_save_torch(data_list, chunk_path)
        print(f"Saved final chunk {chunk_idx} ({len(data_list)} samples) to {chunk_path}")
        total_saved += len(data_list)
    
    print(f"\nTotal saved: {total_saved} triplets in {chunk_idx + 1} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=DEFAULT_ARGS["size"])
    parser.add_argument("--save-path", type=str, default=DEFAULT_ARGS["save_path"])
    parser.add_argument("--workers", type=int, default=DEFAULT_ARGS["workers"])
    args = parser.parse_args()
    
    main(size=args.size, save_path=args.save_path, num_workers=args.workers)
