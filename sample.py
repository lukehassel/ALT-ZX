
import argparse
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pyzx as zx
from tqdm import tqdm

from model import ZXVGAE
from zx_loader import pyzx_to_pyg, qasm_to_pyg, tensor_to_pyzx, pyg_to_pyzx



def decode_and_discretize(model: ZXVGAE, z: torch.Tensor, num_nodes: int,
                          threshold: float = 0.5, device: torch.device = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        z_norm = F.normalize(z, p=2, dim=1)
        adj_probs = torch.sigmoid(torch.mm(z_norm, z_norm.t()))
        adj_matrix = (adj_probs > threshold).float()
        node_type_logits, phase_pred = model.node_phase_decoder(z)
        node_type_probs = torch.softmax(node_type_logits, dim=1)
    
    return adj_matrix, node_type_probs, phase_pred


def optimize_circuit(model: ZXVGAE, target_graph: zx.Graph,
                     num_samples: int = 1000, perturbation_scale: float = 0.1,
                     threshold: float = 0.5, device: torch.device = 'cpu') -> Tuple[zx.Graph, zx.Circuit, int]:
    model.eval()
    
    print("Step A: Encoding target circuit to latent space...")
    target_data = pyzx_to_pyg(target_graph).to(device)
    
    with torch.no_grad():
        mu, logvar = model.encode(target_data.x, target_data.edge_index)
        z_target = model.reparameterize(mu, logvar)
    
    baseline_circuit, baseline_count = extract_circuit_gate_count(target_graph)
    best_graph = target_graph
    best_circuit = baseline_circuit
    best_count = baseline_count
    
    print(f"Baseline gate count: {baseline_count}")
    print(f"\nStep B: Generating {num_samples} perturbed candidates (AltGraph method)...")
    
    valid_count = 0
    improved_count = 0
    
    for i in tqdm(range(num_samples), desc="Sampling candidates"):
        noise = torch.randn_like(z_target) * perturbation_scale
        z_perturbed = z_target + noise
        
        adj_matrix, node_types, phases = decode_and_discretize(
            model, z_perturbed, target_data.num_nodes, threshold, device
        )
        
        try:
            candidate_graph = tensor_to_pyzx(adj_matrix, node_types, phases, threshold)
            
            if check_gflow(candidate_graph):
                valid_count += 1
                
                candidate_circuit, gate_count = extract_circuit_gate_count(candidate_graph)
                
                if gate_count < best_count:
                    best_count = gate_count
                    best_graph = candidate_graph
                    best_circuit = candidate_circuit
                    improved_count += 1
                    print(f"\n✓ Found improved circuit! Gate count: {gate_count} (improvement: {baseline_count - gate_count})")
        
        except Exception as e:
            continue
    
    print(f"\n{'='*60}")
    print("Optimization Summary:")
    print(f"{'='*60}")
    print(f"Total samples: {num_samples}")
    print(f"Valid gflow candidates: {valid_count} ({100*valid_count/num_samples:.1f}%)")
    print(f"Improved circuits found: {improved_count}")
    print(f"Baseline gate count: {baseline_count}")
    print(f"Best gate count: {best_count}")
    if best_count < baseline_count:
        print(f"Improvement: {baseline_count - best_count} gates ({100*(baseline_count-best_count)/baseline_count:.1f}% reduction)")
    else:
        print("No improvement found.")
    print(f"{'='*60}")
    
    return best_graph, best_circuit, best_count


def main():
    parser = argparse.ArgumentParser(
        description='Optimize quantum circuits using ZX-VGAE with AltGraph strategy'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input QASM file or PyZX graph (.pkl)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of latent space samples (N)')
    parser.add_argument('--perturbation_scale', type=float, default=0.1,
                       help='Standard deviation of Gaussian noise (σ)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Edge probability threshold for discretization')
    parser.add_argument('--output', type=str, default='optimized_circuit.qasm',
                       help='Path to save optimized QASM circuit')
    parser.add_argument('--output_graph', type=str, default='optimized_graph.pkl',
                       help='Path to save optimized PyZX graph')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    model_args = checkpoint.get('args', None)
    if model_args:
        in_channels = 6
        model = ZXVGAE(
            in_channels=in_channels,
            hidden_channels=model_args.hidden_dim if hasattr(model_args, 'hidden_dim') else 128,
            latent_dim=model_args.latent_dim if hasattr(model_args, 'latent_dim') else 64,
            num_layers=model_args.num_layers if hasattr(model_args, 'num_layers') else 3
        ).to(args.device)
    else:
        model = ZXVGAE(
            in_channels=6,
            hidden_channels=128,
            latent_dim=64,
            num_layers=3
        ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    
    print(f"\nLoading input: {args.input}")
    input_path = Path(args.input)
    
    if input_path.suffix == '.qasm':
        pyg_data = qasm_to_pyg(str(input_path))
        target_graph = pyg_to_pyzx(pyg_data)
    elif input_path.suffix == '.pkl':
        with open(input_path, 'rb') as f:
            target_graph = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .qasm or .pkl")
    
    print(f"Input graph: {len(target_graph.vertices())} vertices, {len(target_graph.edges())} edges")
    
    print(f"\nStarting optimization with {args.samples} samples...")
    best_graph, best_circuit, best_count = optimize_circuit(
        model, target_graph, args.samples, args.perturbation_scale, 
        args.threshold, args.device
    )
    
    print(f"\nSaving results...")
    
    with open(args.output_graph, 'wb') as f:
        pickle.dump(best_graph, f)
    print(f"Optimized graph saved to {args.output_graph}")
    
    if best_circuit is not None:
        with open(args.output, 'w') as f:
            f.write(best_circuit.to_qasm())
        print(f"Optimized circuit saved to {args.output}")
        print(f"Final gate count: {best_count}")
    else:
        print("Warning: Could not extract circuit from optimized graph.")
    
    print("\nOptimization complete!")


if __name__ == '__main__':
    main()
