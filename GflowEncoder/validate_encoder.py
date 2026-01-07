import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GflowEncoder.model import GraphEncoder, embedding_distance_score
from GflowNet.dataset import create_random_circuit_graph, apply_random_zx_transform, check_gflow
from zx_loader import pyzx_graph_to_pyg
from torch_geometric.data import Batch


def load_trained_encoder(encoder_path='GflowEncoder/encoder.pth', centroid_path='GflowEncoder/valid_centroid.pt'):
    encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64)
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder.eval()
    
    centroid = torch.load(centroid_path, weights_only=True)
    return encoder, centroid


def generate_test_samples(num_samples=100, corruption_levels=[0, 5, 20, 50]):
    samples = {level: [] for level in corruption_levels}
    
    for i in tqdm(range(num_samples), desc="Generating test samples"):
        seed = 100000 + i * 13  # Different seed range than training
        num_qubits = random.randint(3, 6)
        depth = random.randint(5, 15)
        
        graph = create_random_circuit_graph(num_qubits, depth, seed=seed)
        
        for level in corruption_levels:
            corrupted = graph.copy()
            for _ in range(level):
                corrupted = apply_random_zx_transform(corrupted)
            
            data = pyzx_graph_to_pyg(corrupted)
            if data is not None:
                has_gflow = check_gflow(corrupted)
                samples[level].append((data, has_gflow))
    
    return samples


def evaluate_encoder(encoder, centroid, samples):
    results = {}
    
    for level, data_list in samples.items():
        if not data_list:
            continue
        
        scores = []
        gflow_status = []
        
        with torch.no_grad():
            for data, has_gflow in tqdm(data_list, desc=f"Evaluating level {level}"):
                emb = encoder(data.x, data.edge_index)
                similarity = (emb @ centroid.unsqueeze(1)).squeeze().item()
                score = (similarity + 1) / 2  # Map to [0, 1]
                scores.append(score)
                gflow_status.append(has_gflow)
        
        results[level] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'gflow_rate': np.mean(gflow_status),
            'num_samples': len(scores)
        }
    
    return results


def print_results(results):
    print("\n" + "=" * 70)
    print("ENCODER VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Corruption Level':<20} {'Mean Score':<15} {'Std':<10} {'Gflow %':<10} {'N':<10}")
    print("-" * 70)
    
    for level, stats in sorted(results.items()):
        print(f"{level:<20} {stats['mean_score']:.4f}         {stats['std_score']:.4f}     {100*stats['gflow_rate']:.1f}%      {stats['num_samples']}")
    
    print("-" * 70)
    
    levels = sorted(results.keys())
    if len(levels) >= 2:
        score_0 = results[levels[0]]['mean_score']
        score_high = results[levels[-1]]['mean_score']
        
        if score_0 > score_high:
            print("✓ PASS: Valid graphs score higher than corrupted graphs")
            print(f"  Valid: {score_0:.4f}, Corrupted: {score_high:.4f}, Δ = {score_0 - score_high:.4f}")
        else:
            print("✗ FAIL: Corrupted graphs score higher than valid (overfitting?)")
            print(f"  Valid: {score_0:.4f}, Corrupted: {score_high:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("Loading trained encoder...")
    encoder, centroid = load_trained_encoder()
    
    print("\nGenerating held-out test samples...")
    samples = generate_test_samples(num_samples=50, corruption_levels=[0, 5, 10, 20, 50])
    
    print("\nEvaluating encoder...")
    results = evaluate_encoder(encoder, centroid, samples)
    
    print_results(results)
