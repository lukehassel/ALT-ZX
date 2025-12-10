import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ZXreinforce'))

import numpy as np
import torch
import math
from torch_geometric.data import Data
import pyzx as zx
from pyzx.graph.base import BaseGraph
from fractions import Fraction

# Import from ZXreinforce - reuse their conversion functions
from zxreinforce.pyzx_utils import pyzx_to_obs, obs_to_pzx
from zxreinforce.own_constants import (INPUT, OUTPUT, HADAMARD, GREEN, RED, ZERO, 
                                       PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE)


def pyzx_to_pyg(graph: BaseGraph) -> Data:
    obs = pyzx_to_obs(graph)
    colors, angles, _, source, target, _, _, _, _ = obs
    
    n_nodes = len(colors)
    node_types_tensor = torch.tensor(colors, dtype=torch.float32)
    
    phases = []
    for angle in angles:
        if np.all(angle == ZERO):
            phase_val = 0.0
        elif np.all(angle == PI_half):
            phase_val = 0.5
        elif np.all(angle == PI):
            phase_val = 1.0
        elif np.all(angle == PI_three_half):
            phase_val = 1.5
        elif np.all(angle == ARBITRARY):
            phase_val = 0.25
        else:
            phase_val = 0.0
        
        phase_val_normalized = phase_val / 2.0
        phases.append(phase_val_normalized)
    
    phases_tensor = torch.tensor(phases, dtype=torch.float32).unsqueeze(1)
    x = torch.cat([node_types_tensor, phases_tensor], dim=1)
    
    edge_list = []
    for s, t in zip(source, target):
        edge_list.append([int(s), int(t)])
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=n_nodes
    )
    
    data.node_types = node_types_tensor
    data.phases = phases_tensor
    
    return data


def qasm_to_pyg(qasm_file: str) -> Data:
    try:
        import qiskit
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit.from_qasm_file(qasm_file)
        graph = zx.Circuit(circuit).to_graph()
    except ImportError:
        with open(qasm_file, 'r') as f:
            qasm_str = f.read()
        graph = zx.Circuit.from_qasm(qasm_str).to_graph()
    
    return pyzx_to_pyg(graph)


def pyg_to_pyzx(data: Data, threshold: float = 0.5) -> BaseGraph:
    n_nodes = data.num_nodes
    
    node_types = data.node_types if hasattr(data, 'node_types') else data.x[:, :5]
    phases = data.phases if hasattr(data, 'phases') else data.x[:, 5:6]
    
    colors = []
    angles = []
    
    for i in range(n_nodes):
        node_type_vec = node_types[i].cpu().numpy()
        phase_val_normalized = float(phases[i].item())
        phase_val = phase_val_normalized * 2.0
        
        if abs(phase_val) < 0.01:
            angle = ZERO
        elif abs(phase_val - 0.5) < 0.01:
            angle = PI_half
        elif abs(phase_val - 1.0) < 0.01:
            angle = PI
        elif abs(phase_val - 1.5) < 0.01:
            angle = PI_three_half
        else:
            angle = ARBITRARY
        
        colors.append(node_type_vec)
        angles.append(angle)
    
    source = []
    target = []
    if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[:, i]
            source.append(int(src))
            target.append(int(tgt))
    
    obs = (
        np.array(colors),
        np.array(angles),
        np.zeros(len(colors)),
        np.array(source),
        np.array(target),
        np.zeros(len(source)),
        np.array(len(colors)),
        np.array(len(source)),
        np.array([])
    )
    
    return obs_to_pzx(obs)


def tensor_to_pyzx(adj_matrix: torch.Tensor, node_types: torch.Tensor, 
                   phases: torch.Tensor, threshold: float = 0.5) -> BaseGraph:
    n_nodes = node_types.shape[0]
    
    if node_types.shape[1] > 5 or torch.any(node_types < 0):
        node_type_probs = torch.softmax(node_types, dim=1)
    else:
        node_type_probs = node_types
    
    colors = []
    angles = []
    
    for i in range(n_nodes):
        node_type_vec = node_type_probs[i].cpu().numpy()
        phase_val_normalized = float(phases[i].item())
        phase_val = phase_val_normalized * 2.0
        
        if abs(phase_val) < 0.01:
            angle = ZERO
        elif abs(phase_val - 0.5) < 0.01:
            angle = PI_half
        elif abs(phase_val - 1.0) < 0.01:
            angle = PI
        elif abs(phase_val - 1.5) < 0.01:
            angle = PI_three_half
        else:
            angle = ARBITRARY
        
        colors.append(node_type_vec)
        angles.append(angle)
    
    source = []
    target = []
    adj_np = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
    
    if adj_np.ndim == 2 and adj_np.shape[0] == adj_np.shape[1] and adj_np.shape[0] == n_nodes:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj_np[i, j] > threshold or adj_np[j, i] > threshold:
                    source.append(i)
                    target.append(j)
    
    obs = (
        np.array(colors),
        np.array(angles),
        np.zeros(len(colors)),
        np.array(source),
        np.array(target),
        np.zeros(len(source)),
        np.array(len(colors)),
        np.array(len(source)),
        np.array([])
    )
    
    return obs_to_pzx(obs)

