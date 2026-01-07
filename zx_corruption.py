import torch
import random
import numpy as np
import pyzx as zx
from torch_geometric.data import Data
from fractions import Fraction

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zx_loader import pyzx_graph_to_pyg, reconstruct_pyzx_from_6feat


def corrupt_pyzx_graph(graph, num_corruptions=10, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    g = graph.copy()
    vertices = list(g.vertices())
    inputs_set = set(g.inputs())
    outputs_set = set(g.outputs())
    internal = [v for v in vertices if v not in inputs_set and v not in outputs_set]
    
    if len(internal) < 2:
        return g
    
    for _ in range(num_corruptions):
        corruption_type = random.choice(['add_edge', 'remove_edge', 'phase_noise', 'add_edge'])
        
        try:
            if corruption_type == 'add_edge':
                v1, v2 = random.sample(internal, 2)
                if not g.connected(v1, v2):
                    g.add_edge((v1, v2))
                    
            elif corruption_type == 'remove_edge':
                edges = list(g.edges())
                internal_edges = []
                for e in edges:
                    s, t = g.edge_st(e)
                    if s not in inputs_set and s not in outputs_set and \
                       t not in inputs_set and t not in outputs_set:
                        internal_edges.append(e)
                if internal_edges:
                    e = random.choice(internal_edges)
                    g.remove_edge(e)
                    
            elif corruption_type == 'phase_noise':
                zx_spiders = [v for v in internal if g.type(v) in (zx.VertexType.Z, zx.VertexType.X)]
                if zx_spiders:
                    v = random.choice(zx_spiders)
                    noise = Fraction(random.randint(1, 7), 8)
                    g.set_phase(v, g.phase(v) + noise)
        except Exception:
            pass
    
    return g


def corrupt_pyg_graph(pyg_data, 
                      edge_remove_ratio=(0.1, 0.3),
                      edge_add_ratio=(0.05, 0.15),
                      node_remove_ratio=(0.05, 0.15),
                      phase_noise_std=0.1,
                      seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    x = pyg_data.x.clone()
    edge_index = pyg_data.edge_index.clone()
    edge_attr = pyg_data.edge_attr.clone() if pyg_data.edge_attr is not None else None
    
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1] // 2
    
    node_types = x[:, 1]
    boundary_mask = (node_types < 0.5)
    spider_mask = ~boundary_mask
    spider_indices = torch.where(spider_mask)[0].tolist()
    
    # Remove random edges
    remove_ratio = random.uniform(*edge_remove_ratio)
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u < v:
            edge_set.add((u, v))
    
    edges_list = list(edge_set)
    num_to_remove = int(len(edges_list) * remove_ratio)
    
    if num_to_remove > 0 and len(edges_list) > 1:
        edges_to_remove = set(random.sample(edges_list, min(num_to_remove, len(edges_list) - 1)))
        
        keep_mask = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            edge_key = (min(u, v), max(u, v))
            keep_mask.append(edge_key not in edges_to_remove)
        
        keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
        edge_index = edge_index[:, keep_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[keep_mask]
    
    # Add spurious edges
    add_ratio = random.uniform(*edge_add_ratio)
    num_to_add = int(num_edges * add_ratio)
    
    if num_to_add > 0:
        current_edges = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            current_edges.add((min(u, v), max(u, v)))
        
        new_edges = []
        new_attrs = []
        attempts = 0
        max_attempts = num_to_add * 10
        
        while len(new_edges) < num_to_add * 2 and attempts < max_attempts:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v:
                edge_key = (min(u, v), max(u, v))
                if edge_key not in current_edges:
                    current_edges.add(edge_key)
                    new_edges.extend([[u, v], [v, u]])
                    new_attrs.extend([[0.0], [0.0]])
            attempts += 1
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            if edge_attr is not None:
                new_edge_attr = torch.tensor(new_attrs, dtype=torch.float32)
                edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
    
    # Remove random non-boundary nodes
    remove_node_ratio = random.uniform(*node_remove_ratio)
    num_nodes_to_remove = int(len(spider_indices) * remove_node_ratio)
    
    if num_nodes_to_remove > 0 and len(spider_indices) > 1:
        nodes_to_remove = set(random.sample(spider_indices, 
                                            min(num_nodes_to_remove, len(spider_indices) - 1)))
        
        keep_nodes = [i for i in range(num_nodes) if i not in nodes_to_remove]
        old_to_new = {old: new for new, old in enumerate(keep_nodes)}
        
        x = x[keep_nodes]
        
        if edge_index.shape[1] > 0:
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u in nodes_to_remove or v in nodes_to_remove:
                    mask[i] = False
            
            edge_index = edge_index[:, mask]
            if edge_attr is not None:
                edge_attr = edge_attr[mask]
            
            if edge_index.shape[1] > 0:
                new_edge_index = torch.zeros_like(edge_index)
                for i in range(edge_index.shape[1]):
                    new_edge_index[0, i] = old_to_new[edge_index[0, i].item()]
                    new_edge_index[1, i] = old_to_new[edge_index[1, i].item()]
                edge_index = new_edge_index
    
    # Add noise to phases
    if phase_noise_std > 0:
        phase_noise = torch.randn(x.shape[0]) * phase_noise_std
        x[:, 4] = x[:, 4] + phase_noise
    
    # Update VertexID and Degree
    x[:, 0] = torch.arange(x.shape[0], dtype=torch.float32)
    if edge_index.shape[1] > 0:
        degrees = torch.zeros(x.shape[0])
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i].item()] += 0.5
        x[:, 3] = degrees
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    print("Testing corruption utilities...")
    
    x = torch.randn(10, 6)
    x[:, 1] = torch.tensor([0, 1, 2, 1, 2, 1, 2, 1, 0, 0], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    corrupted = corrupt_pyg_graph(data, seed=42)
    print(f"Original: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Corrupted: {corrupted.x.shape[0]} nodes, {corrupted.edge_index.shape[1]} edges")
    print("✓ PyG corruption test passed")
