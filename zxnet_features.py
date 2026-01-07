import torch
import pyzx as zx
from pyzx.utils import VertexType, EdgeType
from torch_geometric.data import Data
from fractions import Fraction
from zx_loader import circuit_to_pyg, pyg_to_circuit


def extract_zxnet_features(zx_graph: zx.Graph) -> Data:
    zx.full_reduce(zx_graph) 
    
    def get_degree(g, v):
        if hasattr(g, "degree"): return g.degree(v)
        if hasattr(g, "valence"): return g.valence(v)
        if hasattr(g, "neighbors"): return len(list(g.neighbors(v)))
        return 0

    node_mapping = {node: i for i, node in enumerate(zx_graph.vertices())}
    num_nodes = len(node_mapping)
    
    rows = zx_graph.rows()
    qubits = zx_graph.qubits()
    phases = zx_graph.phases()
    types = zx_graph.types()
    
    # Features: [VertexID, NodeType, Row, Degree, Phase, Qubit]
    x_features = []

    for v in zx_graph.vertices():
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
        feat_degree = float(get_degree(zx_graph, v))
        feat_phase = float(phases.get(v, 0))
        feat_qubit = float(qubits.get(v, -1))
        
        x_features.append([feat_id, feat_type, feat_row, feat_degree, feat_phase, feat_qubit])

    x = torch.tensor(x_features, dtype=torch.float32)

    edge_index_list = []
    edge_attr_list = []

    for e in zx_graph.edges():
        src, tgt = zx_graph.edge_s(e), zx_graph.edge_t(e)
        u, v = node_mapping[src], node_mapping[tgt]
        
        e_type = zx_graph.edge_type(e)
        feat_edge_type = 1.0 if e_type == EdgeType.HADAMARD else 0.0
        
        edge_index_list.append([u, v])
        edge_attr_list.append([feat_edge_type])
        
        edge_index_list.append([v, u])
        edge_attr_list.append([feat_edge_type])

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    print("Generating test circuit...")
    try:
        circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=3, depth=5)
        g = circ.to_graph()
        
        data = extract_zxnet_features(g)
        
        print("Node Features (x):", data.x.shape)
        print("Edge Attributes:", data.edge_attr.shape)
        print("Example Node (ID, Type, Row, Deg, Phase, Qubit):")
        print(data.x[0])
        print("\nSuccess!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


import qiskit
from qiskit import qasm2


def pyg_to_qiskit_print(data: Data):
    try:
        qc = pyg_to_circuit(data)
        print(qc)
        return qc
    except Exception as e:
        print(f"Failed to convert or print: {e}")
