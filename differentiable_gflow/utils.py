import torch
import random
from gflow_algorithm import find_gflow

def local_complementation(adj, v):
    """Apply local complementation: toggles edges between neighbors of v."""
    neighbors = (adj[v] > 0.5).float()
    # Toggle sub-graph induced by neighbors (excluding diagonal)
    toggle_grid = torch.outer(neighbors, neighbors)
    new_adj = (adj + toggle_grid) % 2.0
    return new_adj.fill_diagonal_(0)

def generate_random_gflow_graphs(n, n_graphs, p=0.2):
    """Generates a batch of graphs with valid GFlow and random I/O."""
    results = []
    while len(results) < n_graphs:
        adj = torch.triu((torch.rand(n, n) < p).float(), diagonal=1)
        adj = adj + adj.T
        
        idx = torch.randperm(n)
        n_io = random.randint(1, n // 2)
        inputs, outputs = sorted(idx[:n_io].tolist()), sorted(idx[n_io:2*n_io].tolist())
        
        edges = adj.nonzero().tolist()
        if find_gflow(n, edges, inputs, outputs):
            results.append((adj, inputs, outputs))
            if len(results) % 10 == 0:
                print(f"  Generated {len(results)}/{n_graphs} starting graphs")
    return results

def apply_random_zx_rules(adj, n_rules):
    """Apply n random local complementations."""
    for _ in range(n_rules):
        adj = local_complementation(adj, random.randint(0, adj.shape[0] - 1))
    return adj

def apply_random_transformation(n, adj, inputs, outputs, step):
    """Apply random perturbation (fade in/out) or I/O shift."""
    adj, inputs, outputs = adj.clone(), list(inputs), list(outputs)
    u_idx, v_idx = torch.triu_indices(n, n, offset=1)
    
    if step % 5 <= 3: # Fade edges: decrease or increase edge weights
        # If <=1: fade out existing; if 2,3: fade in random
        mask = (adj[u_idx, v_idx] > 0.01) if step % 5 <= 1 else torch.ones_like(u_idx, dtype=torch.bool)
        if mask.any():
            i = random.choice(mask.nonzero().flatten().tolist())
            u, v = u_idx[i], v_idx[i]
            delta = 0.2 if step % 5 > 1 else -0.2
            adj[u, v] = adj[v, u] = torch.clamp(adj[u, v] + delta, 0, 1)
    else: # I/O shift
        if len(outputs) > 1 and random.random() < 0.5:
            outputs.pop(random.randint(0, len(outputs)-1))
        elif (candidates := [i for i in range(n) if i not in inputs + outputs]):
            inputs.append(random.choice(candidates))
            
    return adj, inputs, outputs
