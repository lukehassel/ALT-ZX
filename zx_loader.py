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
from qiskit import qasm2

# Import from ZXreinforce - reuse their conversion functions
from zxreinforce.pyzx_utils import pyzx_to_obs, obs_to_pzx
from zxreinforce.own_constants import (INPUT, OUTPUT, HADAMARD, GREEN, RED, ZERO, 
                                       PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE)
from pyzx.utils import VertexType, EdgeType


def graph_degrees(graph: BaseGraph):
    """Best-effort degree computation that works across PyZX graph variants."""
    return {
        v: (
            (graph.degree(v) if hasattr(graph, "degree") else None)
            or (graph.valence(v) if hasattr(graph, "valence") else None)
            or (len(list(graph.neighbors(v))) if hasattr(graph, "neighbors") else None)
            or (len(list(graph.adjacent(v))) if hasattr(graph, "adjacent") else 0)
        )
        for v in graph.vertices()
    }


def circuit_to_pyg(qc) -> Data:
    """
    pyg format from pytorch for Graph Neural Networks
    """
    
    qc = qc.remove_final_measurements(inplace=False)
    
    # Convert Qiskit -> PyZX Graph
    graph = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_graph()
    
    # Organize data for fast lookup
    rows = graph.rows()
    qubits = graph.qubits()
    phases = graph.phases()
    types = graph.types()
    degrees = graph_degrees(graph)  # ZXNet uses degree as a feature [cite: 141]
    
    # Map graph vertex indices to 0..N-1 tensor indices
    node_mapping = {node: i for i, node in enumerate(graph.vertices())}
    
    node_features = []
    
    for v in graph.vertices():
        t = types[v]
        is_boundary = 1 if t == VertexType.BOUNDARY else 0
        is_z = 1 if t == VertexType.Z else 0
        is_x = 1 if t == VertexType.X else 0
        phase = float(phases[v]) 
        row_index = float(rows.get(v, -1))
        qubit_index = float(qubits.get(v, -1))
        degree = float(degrees.get(v, 0))

        feat = [is_boundary, is_z, is_x, phase, row_index, qubit_index, degree]
        node_features.append(feat)

    # convert feature list to tensor
    x = torch.tensor(node_features, dtype=torch.float32)

    edge_indices = []
    edge_attrs = []
    
    for e in graph.edges():
        src, tgt = e
        e_type = graph.edge_type(e)
        
        u, v_idx = node_mapping[src], node_mapping[tgt]
        
        is_hadamard = 1.0 if e_type == EdgeType.HADAMARD else 0.0
        
        # Add undirected edges (u->v and v->u)
        edge_indices.append([u, v_idx])
        edge_indices.append([v_idx, u])
        
        # Edge attributes must match edge_index length. That is why its added twice.
        edge_attrs.append([is_hadamard])
        edge_attrs.append([is_hadamard])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # 4. Create Data Object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Wrapper to maintain compatibility with your existing loading
def zxnet_qasm_to_pyg(qasm_file: str) -> Data:
    try:
        # Try loading directly via PyZX (lighter weight)
        circ = zx.Circuit.from_qasm(qasm_file)
    except Exception:
        # Fallback to reading file manually if path string
        with open(qasm_file, 'r') as f:
            qasm_str = f.read()
        circ = zx.Circuit.from_qasm(qasm_str)
        
    graph = circ.to_graph()
    return zxnet_pyzx_to_pyg(graph)

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

if __name__ == "__main__":
    
    print("--- Generating Circuit ---")
    circ = zx.Circuit(qubit_amount=2)
    circ.add_gate("H", 0)       # Hadamard on qubit 0
    circ.add_gate("CNOT", 0, 1) # CNOT between 0 and 1

    # Convert circuit to a graph
    graph = circ.to_graph()
    print(f"Original Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

    # --- 3. Run the Conversion ---
    print("\n--- Converting to PyG ---")
    data = pyzx_to_pyg(graph)
    
    print(data)

