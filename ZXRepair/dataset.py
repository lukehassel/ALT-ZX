import os
import sys
import random
import torch
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

import pyzx as zx
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuit_utils import create_random_circuit_with_universal_gates
from zx_loader import pyzx_graph_to_pyg
from zx_corruption import corrupt_pyg_graph


CONFIG = {
    'num_samples': 50000,
    'num_qubits_range': (3, 10),
    'depth_range': (5, 20),
    'max_nodes': 256,
    'output_path': '/work/wo057552/repair_dataset.pt',
    'num_workers': 8,
    'edge_remove_ratio': (0.1, 0.3),
    'edge_add_ratio': (0.05, 0.15),
    'node_remove_ratio': (0.05, 0.15),
    'phase_noise_std': 0.1,
}


def generate_single_pair(idx, config):
    seed = config.get('base_seed', 42) + idx
    random.seed(seed)
    
    num_qubits = random.randint(*config['num_qubits_range'])
    depth = random.randint(*config['depth_range'])
    
    try:
        qc = create_random_circuit_with_universal_gates(num_qubits, depth, seed=seed)
        
        from qiskit import qasm2
        zx_circuit = zx.Circuit.from_qasm(qasm2.dumps(qc))
        zx_graph = zx_circuit.to_graph()
        
        if zx_graph.num_vertices() > config['max_nodes']:
            return None
        
        original = pyzx_graph_to_pyg(zx_graph, max_nodes=None)
        
        if original is None or original.x.shape[0] < 3:
            return None
        
        corrupted = corrupt_pyg_graph(
            original,
            edge_remove_ratio=config['edge_remove_ratio'],
            edge_add_ratio=config['edge_add_ratio'],
            node_remove_ratio=config['node_remove_ratio'],
            phase_noise_std=config['phase_noise_std'],
            seed=seed + 1000000
        )
        
        return pickle.dumps((corrupted, original))
        
    except Exception as e:
        return None


def generate_repair_dataset(num_samples, config=None, num_workers=None, verbose=True):
    if config is None:
        config = CONFIG.copy()
    
    if num_workers is None:
        num_workers = min(cpu_count(), config.get('num_workers', 8))
    
    config['base_seed'] = random.randint(0, 1000000)
    
    if verbose:
        print(f"Generating {num_samples} repair dataset pairs...")
        print(f"  Qubits: {config['num_qubits_range']}")
        print(f"  Depth: {config['depth_range']}")
        print(f"  Edge remove ratio: {config['edge_remove_ratio']}")
        print(f"  Edge add ratio: {config['edge_add_ratio']}")
        print(f"  Node remove ratio: {config['node_remove_ratio']}")
        print(f"  Workers: {num_workers}")
    
    indices = list(range(num_samples * 2))
    
    worker_fn = partial(generate_single_pair, config=config)
    
    checkpoint_interval = 5000
    checkpoint_path = '/work/wo057552/zxrepair_checkpoint.pt'
    
    pairs = []
    last_checkpoint = 0
    
    if num_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(num_workers) as pool:
            for result in pool.imap_unordered(worker_fn, indices, chunksize=100):
                if result is not None:
                    pair = pickle.loads(result)
                    pairs.append(pair)
                    
                    if verbose and len(pairs) % 1000 == 0:
                        print(f"  Progress: {len(pairs)}/{num_samples}")
                    
                    if len(pairs) - last_checkpoint >= checkpoint_interval:
                        try:
                            with open(checkpoint_path, 'wb') as f:
                                pickle.dump(pairs, f, protocol=4)
                            print(f"  Checkpoint saved: {len(pairs)} pairs to {checkpoint_path}")
                            last_checkpoint = len(pairs)
                        except Exception as e:
                            print(f"  Warning: checkpoint save failed: {e}")
                    
                if len(pairs) >= num_samples:
                    pool.terminate()
                    break
    else:
        for idx in indices:
            result = generate_single_pair(idx, config)
            if result is not None:
                pair = pickle.loads(result)
                pairs.append(pair)
                
                if len(pairs) - last_checkpoint >= checkpoint_interval:
                    try:
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump(pairs, f, protocol=4)
                        print(f"  Checkpoint saved: {len(pairs)} pairs to {checkpoint_path}")
                        last_checkpoint = len(pairs)
                    except Exception as e:
                        print(f"  Warning: checkpoint save failed: {e}")
                        
            if len(pairs) >= num_samples:
                break
            if verbose and len(pairs) % 1000 == 0:
                print(f"  Progress: {len(pairs)}/{num_samples}")
    
    pairs = pairs[:num_samples]
    
    if verbose:
        print(f"Generated {len(pairs)} pairs")
        
        orig_nodes = [p[1].x.shape[0] for p in pairs]
        corr_nodes = [p[0].x.shape[0] for p in pairs]
        print(f"  Original nodes: min={min(orig_nodes)}, max={max(orig_nodes)}, avg={sum(orig_nodes)/len(orig_nodes):.1f}")
        print(f"  Corrupted nodes: min={min(corr_nodes)}, max={max(corr_nodes)}, avg={sum(corr_nodes)/len(corr_nodes):.1f}")
    
    return pairs


def save_dataset(pairs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"Saved {len(pairs)} pairs to {output_path}")


def load_dataset(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return torch.load(filepath, weights_only=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ZXRepair training dataset')
    parser.add_argument('--num_samples', type=int, default=CONFIG['num_samples'],
                        help='Number of pairs to generate')
    parser.add_argument('--output', type=str, default=CONFIG['output_path'],
                        help='Output file path')
    parser.add_argument('--workers', type=int, default=CONFIG['num_workers'],
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    pairs = generate_repair_dataset(
        num_samples=args.num_samples,
        num_workers=args.workers,
        verbose=True
    )
    
    save_dataset(pairs, args.output)
