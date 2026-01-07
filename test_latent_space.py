import torch
import pyzx as zx
import numpy as np
from qiskit import QuantumCircuit
from qiskit import qasm2

from GenZX.model import GraphVAE
from GenZX.data_loader import GraphVAEDataset


def circuit_to_zx_graph(qc):
    qasm_str = qasm2.dumps(qc)
    circuit = zx.Circuit.from_qasm(qasm_str)
    graph = circuit.to_graph()
    return graph


def zx_to_pyg_data(graph, max_nodes=64):
    from torch_geometric.data import Data
    
    nodes = list(graph.vertices())
    n_nodes = min(len(nodes), max_nodes)
    node_map = {v: i for i, v in enumerate(nodes[:n_nodes])}
    
    x = torch.zeros(n_nodes, 6)
    for i, v in enumerate(nodes[:n_nodes]):
        vtype = graph.type(v)
        if vtype == zx.VertexType.Z:
            x[i, 0] = 1.0
        elif vtype == zx.VertexType.X:
            x[i, 1] = 1.0
        elif vtype == zx.VertexType.BOUNDARY:
            x[i, 2] = 1.0
        elif vtype == zx.VertexType.H_BOX:
            x[i, 3] = 1.0
        
        phase = graph.phase(v)
        if hasattr(phase, 'real'):
            x[i, 4] = float(phase.real) / np.pi
        else:
            x[i, 4] = float(phase) / np.pi
    
    edges = []
    for e in graph.edges():
        s, t = graph.edge_st(e)
        if s in node_map and t in node_map:
            edges.append([node_map[s], node_map[t]])
            edges.append([node_map[t], node_map[s]])
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


def get_latent_embedding(model, pyg_data, max_nodes=64):
    model.eval()
    
    dataset = GraphVAEDataset([pyg_data], max_nodes=max_nodes)
    data = dataset[0]
    
    with torch.no_grad():
        node_features = data['node_features'].unsqueeze(0).float()
        adj = data['adj'].unsqueeze(0).float()
        
        if torch.cuda.is_available():
            node_features = node_features.cuda()
            adj = adj.cuda()
        
        z_mu, _ = model.vae.encode(node_features, adj)
        
    return z_mu.cpu().squeeze()


def create_equivalent_circuits():
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.h(0)
    qc1.h(1)
    
    qc2 = QuantumCircuit(2)
    qc2.h(1)
    qc2.cz(0, 1)
    qc2.h(1)
    
    return qc1, qc2


def create_non_equivalent_circuits():
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.h(1)
    
    qc2 = QuantumCircuit(2)
    qc2.x(0)
    qc2.x(1)
    
    return qc1, qc2


def main():
    print("=" * 60)
    print("LATENT SPACE DISTANCE TEST")
    print("=" * 60)
    
    max_nodes = 64
    
    print("\nLoading model...")
    model = GraphVAE(
        input_dim=max_nodes,
        hidden_dim=64,
        latent_dim=256,
        max_num_nodes=max_nodes,
        pool='sum',
        lambda_zxnet=1.0
    )
    
    try:
        checkpoint = torch.load('GenZX/checkpoints/model.pth', 
                                map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("WARNING: No checkpoint found, using random model")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("\n" + "=" * 60)
    print("TEST 1: Functionally EQUIVALENT circuits")
    print("=" * 60)
    
    qc1, qc2 = create_equivalent_circuits()
    print(f"\nCircuit 1 (H-CNOT-H pattern):")
    print(qc1)
    print(f"\nCircuit 2 (CZ equivalent):")
    print(qc2)
    
    zx1 = circuit_to_zx_graph(qc1)
    zx2 = circuit_to_zx_graph(qc2)
    print(f"\nZX Graph 1: {zx1.num_vertices()} vertices, {zx1.num_edges()} edges")
    print(f"ZX Graph 2: {zx2.num_vertices()} vertices, {zx2.num_edges()} edges")
    
    pyg1 = zx_to_pyg_data(zx1, max_nodes)
    pyg2 = zx_to_pyg_data(zx2, max_nodes)
    
    z1 = get_latent_embedding(model, pyg1, max_nodes)
    z2 = get_latent_embedding(model, pyg2, max_nodes)
    
    l2_dist_equiv = torch.norm(z1 - z2).item()
    cosine_sim_equiv = torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
    
    print(f"\nLatent embeddings:")
    print(f"  z1 shape: {z1.shape}, norm: {torch.norm(z1):.4f}")
    print(f"  z2 shape: {z2.shape}, norm: {torch.norm(z2):.4f}")
    print(f"\nDistances (EQUIVALENT circuits - should be SMALL):")
    print(f"  L2 distance: {l2_dist_equiv:.4f}")
    print(f"  Cosine similarity: {cosine_sim_equiv:.4f}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Functionally NON-EQUIVALENT circuits")
    print("=" * 60)
    
    qc3, qc4 = create_non_equivalent_circuits()
    print(f"\nCircuit 3 (H gates):")
    print(qc3)
    print(f"\nCircuit 4 (X gates):")
    print(qc4)
    
    zx3 = circuit_to_zx_graph(qc3)
    zx4 = circuit_to_zx_graph(qc4)
    print(f"\nZX Graph 3: {zx3.num_vertices()} vertices, {zx3.num_edges()} edges")
    print(f"ZX Graph 4: {zx4.num_vertices()} vertices, {zx4.num_edges()} edges")
    
    pyg3 = zx_to_pyg_data(zx3, max_nodes)
    pyg4 = zx_to_pyg_data(zx4, max_nodes)
    
    z3 = get_latent_embedding(model, pyg3, max_nodes)
    z4 = get_latent_embedding(model, pyg4, max_nodes)
    
    l2_dist_nonequiv = torch.norm(z3 - z4).item()
    cosine_sim_nonequiv = torch.nn.functional.cosine_similarity(z3.unsqueeze(0), z4.unsqueeze(0)).item()
    
    print(f"\nLatent embeddings:")
    print(f"  z3 shape: {z3.shape}, norm: {torch.norm(z3):.4f}")
    print(f"  z4 shape: {z4.shape}, norm: {torch.norm(z4):.4f}")
    print(f"\nDistances (NON-EQUIVALENT circuits - should be LARGE):")
    print(f"  L2 distance: {l2_dist_nonequiv:.4f}")
    print(f"  Cosine similarity: {cosine_sim_nonequiv:.4f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n                         L2 Distance    Cosine Sim")
    print(f"  Equivalent circuits:   {l2_dist_equiv:10.4f}    {cosine_sim_equiv:10.4f}")
    print(f"  Non-equivalent:        {l2_dist_nonequiv:10.4f}    {cosine_sim_nonequiv:10.4f}")
    print(f"\n  Ratio (non-eq/eq):     {l2_dist_nonequiv/l2_dist_equiv:.2f}x")
    
    if l2_dist_nonequiv > l2_dist_equiv:
        print("\n  ✅ Non-equivalent circuits ARE farther apart in latent space")
    else:
        print("\n  ❌ Non-equivalent circuits are NOT farther apart - model not learning semantics")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
