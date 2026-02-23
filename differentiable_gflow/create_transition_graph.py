import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple
from gflow_algorithm import find_gflow
from loss_function import GFlowLoss
from utils import apply_random_zx_rules, generate_random_gflow_graphs, apply_random_transformation

def get_data_points(n_graphs: int = 1000, n_steps: int = 50, graph_size: int = 20):
    print(f"Running experiment v2: {n_graphs} graphs, {n_steps} steps")
    
    all_steps = []
    all_losses = []
    all_colors = []  # Green = has gflow, red = no gflow
    
    model_cache = {}

    # Pre-generate n_graphs with valid GFlow and diverse I/O configurations
    starting_graphs = generate_random_gflow_graphs(graph_size, n_graphs, p=0.2)

    for g_idx, (adj, inputs, outputs) in enumerate(starting_graphs):
        n = graph_size
        current_inputs = list(inputs)
        current_outputs = list(outputs)
        
        for step in range(n_steps):
            # Check discrete gflow on the current (possibly perturbed) graph
            threshold = 0.5
            binary_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j].item() > threshold:
                        binary_edges.append((i, j))
            
            gflow_obj = find_gflow(n, binary_edges, current_inputs, current_outputs)
            has_gflow = gflow_obj is not None
            
            # Get loss
            key = (tuple(sorted(current_inputs)), tuple(sorted(current_outputs)))
            if key not in model_cache:
                model_cache[key] = GFlowLoss(n, current_inputs, current_outputs, inner_iterations=100)
            
            model = model_cache[key]
            
            with torch.no_grad():
                loss = model(adj).item()
            
            all_steps.append(step)
            all_losses.append(min(loss, 1.0))
            all_colors.append('green' if has_gflow else 'red')
            
            # Apply random transformation: gradually perturb adj
            adj, current_inputs, current_outputs = apply_random_transformation(
                n, adj, current_inputs, current_outputs, step
            )
            
    return np.array(all_steps), np.array(all_losses), all_colors


def create_scatter_visualization(steps, losses, colors, output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    green_mask = [c == 'green' for c in colors]
    red_mask = [c == 'red' for c in colors]
    
    ax.scatter(steps[green_mask], losses[green_mask], c='green', alpha=0.3, s=5, label='Has GFlow')
    ax.scatter(steps[red_mask], losses[red_mask], c='red', alpha=0.3, s=5, label='No GFlow')
    
    ax.set_xlabel('Transformation Step', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('(Green = has gflow, Red = no gflow)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.03, color='blue', linestyle='--', alpha=0.5, label='Classification threshold (0.03)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    steps, losses, colors = get_data_points(n_graphs=100, n_steps=30, graph_size=10)
    
    create_scatter_visualization(
        steps, losses, colors,
        "gflow_transition_scatter.png"
    )
