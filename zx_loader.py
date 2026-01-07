import numpy as np
import torch
import math
from torch_geometric.data import Data
import pyzx as zx
from pyzx.graph.base import BaseGraph
from fractions import Fraction
import qiskit
from qiskit import qasm2

from zxreinforce_utils.pyzx_utils import pyzx_to_obs, obs_to_pzx
from zxreinforce_utils.own_constants import (INPUT, OUTPUT, HADAMARD, GREEN, RED, ZERO, 
                                             PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE)
from pyzx.utils import VertexType, EdgeType


def circuit_to_pyg(qc) -> Data:
    qc = qc.remove_final_measurements(inplace=False)
    graph = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_graph()
    
    rows = graph.rows()
    qubits = graph.qubits()
    phases = graph.phases()
    types = graph.types()
    
    def get_degree(g, v):
        if hasattr(g, "degree"): return g.degree(v)
        if hasattr(g, "valence"): return g.valence(v)
        if hasattr(g, "neighbors"): return len(list(g.neighbors(v)))
        return 0
    
    node_mapping = {node: i for i, node in enumerate(graph.vertices())}
    
    # Features: [VertexID, NodeType, Row, Degree, Phase, Qubit]
    node_features = []
    
    for v in graph.vertices():
        feat_id = float(node_mapping[v])
        
        t = types[v]
        if t == VertexType.BOUNDARY:
            feat_type = 0.0
        elif t == VertexType.Z:
            feat_type = 1.0
        elif t == VertexType.X:
            feat_type = 2.0
        else:
            feat_type = -1.0
            
        feat_row = float(rows.get(v, -1))
        feat_degree = float(get_degree(graph, v))
        feat_phase = float(phases.get(v, 0))
        feat_qubit = float(qubits.get(v, -1))

        feat = [feat_id, feat_type, feat_row, feat_degree, feat_phase, feat_qubit]
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float32)

    edge_indices = []
    edge_attrs = []
    
    for e in graph.edges():
        src, tgt = e
        e_type = graph.edge_type(e)
        
        u, v_idx = node_mapping[src], node_mapping[tgt]
        
        edge_weight = 0.5 if e_type == EdgeType.HADAMARD else 1.0
        
        edge_indices.append([u, v_idx])
        edge_indices.append([v_idx, u])
        
        edge_attrs.append([edge_weight])
        edge_attrs.append([edge_weight])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def pyzx_graph_to_pyg(graph, max_nodes=None) -> Data:
    vertices = list(graph.vertices())
    n_nodes = len(vertices) if max_nodes is None else min(len(vertices), max_nodes)
    
    if n_nodes == 0:
        return None
    
    node_map = {v: i for i, v in enumerate(vertices[:n_nodes])}
    
    rows = graph.rows()
    qubits = graph.qubits()
    phases = graph.phases()
    types = graph.types()
    
    def get_degree(g, v):
        if hasattr(g, "degree"): return g.degree(v)
        if hasattr(g, "valence"): return g.valence(v)
        if hasattr(g, "neighbors"): return len(list(g.neighbors(v)))
        return 0
    
    # Features: [VertexID, NodeType, Row, Degree, Phase, Qubit]
    node_features = []
    for v in vertices[:n_nodes]:
        feat_id = float(node_map[v])
        
        t = types[v]
        if t == VertexType.BOUNDARY:
            feat_type = 0.0
        elif t == VertexType.Z:
            feat_type = 1.0
        elif t == VertexType.X:
            feat_type = 2.0
        else:
            feat_type = -1.0
        
        feat_row = float(rows.get(v, -1))
        feat_degree = float(get_degree(graph, v))
        feat_phase = float(phases.get(v, 0))
        feat_qubit = float(qubits.get(v, -1))
        
        node_features.append([feat_id, feat_type, feat_row, feat_degree, feat_phase, feat_qubit])
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    edge_indices = []
    edge_attrs = []
    
    for e in graph.edges():
        s, t = graph.edge_st(e)
        if s in node_map and t in node_map:
            u, v_idx = node_map[s], node_map[t]
            e_type = graph.edge_type(e)
            edge_weight = 0.5 if e_type == EdgeType.HADAMARD else 1.0
            
            edge_indices.append([u, v_idx])
            edge_indices.append([v_idx, u])
            edge_attrs.append([edge_weight])
            edge_attrs.append([edge_weight])
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)


def zxnet_qasm_to_pyg(qasm_file: str) -> Data:
    try:
        circ = zx.Circuit.from_qasm(qasm_file)
    except Exception:
        with open(qasm_file, 'r') as f:
            qasm_str = f.read()
        circ = zx.Circuit.from_qasm(qasm_str)
        
    return circuit_to_pyg(qiskit.QuantumCircuit.from_qasm_str(circ.to_qasm()))


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


def reconstruct_pyzx_from_6feat(data: Data) -> zx.Graph:
    g = zx.Graph()
    
    num_nodes = data.x.shape[0]
    idx_to_v = {}
    
    for i in range(num_nodes):
        feat = data.x[i]
        type_val = feat[1]
        row = int(float(feat[2]))
        phase_val = float(feat[4])
        qubit = int(float(feat[5]))
        
        phase_frac = Fraction(phase_val).limit_denominator(1000000)
        
        if abs(type_val - 0.0) < 0.1:
            ty = VertexType.BOUNDARY
        elif abs(type_val - 1.0) < 0.1:
            ty = VertexType.Z
        elif abs(type_val - 2.0) < 0.1:
            ty = VertexType.X
        else:
            ty = VertexType.Z
        
        v = g.add_vertex(ty, row=row, qubit=qubit, phase=phase_frac)
        idx_to_v[i] = v
        
    if data.edge_index.numel() > 0:
        edges = data.edge_index.t().tolist()
        attrs = data.edge_attr
        
        seen_edges = set()
        
        for k, (u_idx, v_idx) in enumerate(edges):
            if u_idx > v_idx: continue 
            
            if (u_idx, v_idx) in seen_edges: continue
            seen_edges.add((u_idx, v_idx))
            
            u = idx_to_v[u_idx]
            v = idx_to_v[v_idx]
            
            is_hadamard = attrs[k][0] > 0.5
            ety = EdgeType.HADAMARD if is_hadamard else EdgeType.SIMPLE
            
            g.add_edge((u, v), ety)
            
    inputs = []
    outputs = []
    
    boundaries = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
    
    if boundaries:
        rows = {v: g.row(v) for v in boundaries}
        min_row = min(rows.values())
        max_row = max(rows.values())
        
        inputs = sorted([v for v in boundaries if rows[v] == min_row], key=lambda v: g.qubit(v))
        outputs = sorted([v for v in boundaries if rows[v] == max_row], key=lambda v: g.qubit(v))

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))
    
    return g


def pyg_to_circuit(data: Data):
    if data.x.shape[1] != 6:
        raise ValueError(f"Expected 6 node features, got {data.x.shape[1]}")
        
    g = reconstruct_pyzx_from_6feat(data)
    
    try:
        circ = zx.Circuit.from_graph(g)
    except TypeError:
        circ = zx.extract_circuit(g.copy())

    qasm_str = circ.to_qasm()
    
    try:
        qc = qasm2.loads(qasm_str)
    except Exception:
        qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        
    return qc


if __name__ == "__main__":
    print("--- Generating Circuit ---")
    circ = zx.Circuit(qubit_amount=2)
    circ.add_gate("H", 0)
    circ.add_gate("CNOT", 0, 1)

    graph = circ.to_graph()
    print(f"Original Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")

    print("\n--- Converting to PyG ---")
    data = pyzx_graph_to_pyg(graph)
    
    print(data)
