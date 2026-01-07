"""Debug GenZX extraction failures."""
import torch
import pyzx as zx
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZX.model import GraphVAE
from GenZX.data_loader import GraphVAEDataset
from GenZX.dataset import create_random_circuit_graph
from zx_loader import pyzx_graph_to_pyg, reconstruct_pyzx_from_6feat
from torch_geometric.data import Data


def adj_to_zx_graph(adj_matrix, node_features, threshold=0.5):
    if adj_matrix.is_cuda:
        adj_matrix = adj_matrix.cpu()
    if node_features.is_cuda:
        node_features = node_features.cpu()
    
    feature_mask = node_features.sum(dim=1) > 0.1
    binary_adj = (adj_matrix > threshold).float()
    degree = binary_adj.sum(dim=1)
    connectivity_mask = degree > 0
    active_mask = feature_mask & connectivity_mask
    active_indices = torch.where(active_mask)[0].tolist()
    
    if not active_indices:
        return None, "No active nodes"
    
    active_node_features = node_features[active_indices]
    
    src_list, dst_list, edge_attr_list = [], [], []
    for i, old_i in enumerate(active_indices):
        for j, old_j in enumerate(active_indices):
            if i >= j:
                continue
            if adj_matrix[old_i, old_j] > threshold:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_attr_list.extend([[0.0], [0.0]])
    
    if not src_list:
        return None, "No edges above threshold"
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    data = Data(x=active_node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    try:
        graph = reconstruct_pyzx_from_6feat(data)
        return graph, None
    except Exception as e:
        return None, f"reconstruct_pyzx_from_6feat failed: {type(e).__name__}: {e}"


def main():
    print("="*60)
    print("GenZX Extraction Debug")
    print("="*60)
    
    # Load model
    model = GraphVAE(
        input_dim=64,
        hidden_dim=64,
        latent_dim=1024,
        max_num_nodes=64,
        pool='sum',
        lambda_zxnet=1.0
    ).cuda()
    
    checkpoint = torch.load('GenZX/checkpoints/model.pth', map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded\n")
    
    # Generate a simple test graph
    print("Generating test input graph...")
    zx_graph = create_random_circuit_graph(num_qubits=3, depth=3, seed=42)
    print(f"  Input ZX graph: {zx_graph.num_vertices()} vertices, {zx_graph.num_edges()} edges")
    
    # Check if input extracts
    try:
        input_circuit = zx.extract_circuit(zx_graph.copy())
        print(f"  Input extracts: YES ({input_circuit.qubits} qubits)")
    except Exception as e:
        print(f"  Input extracts: NO ({type(e).__name__})")
    
    # Convert to PyG
    pyg_data = pyzx_graph_to_pyg(zx_graph)
    print(f"  PyG: {pyg_data.x.shape[0]} nodes, {pyg_data.edge_index.shape[1]} edges")
    
    # Process through model
    dataset = GraphVAEDataset([pyg_data], max_nodes=64)
    input_data = dataset[0]
    
    print("\n" + "="*60)
    print("Model Forward Pass")
    print("="*60)
    
    with torch.no_grad():
        node_features = input_data['node_features'].unsqueeze(0).float().cuda()
        adj = input_data['adj'].unsqueeze(0).float().cuda()
        
        print(f"Input adj shape: {adj.shape}")
        print(f"Input adj stats: mean={adj.mean():.4f}, edges>0.5={int((adj > 0.5).sum().item())}")
        
        # Encode
        z_mu, z_logvar = model.vae.encode(node_features, adj)
        print(f"\nLatent z_mu: shape={z_mu.shape}, range=[{z_mu.min():.3f}, {z_mu.max():.3f}]")
        
        # Decode with ZERO noise
        decoded = model.vae.decode(z_mu)
        out = torch.sigmoid(decoded)
        recon_adj = model.recover_adj_differentiable_batch(out)
        
        print(f"\nReconstructed adj shape: {recon_adj.shape}")
        print(f"Recon adj stats: mean={recon_adj.mean():.4f}, range=[{recon_adj.min():.4f}, {recon_adj.max():.4f}]")
        print(f"Edges > 0.5: {int((recon_adj > 0.5).sum().item())}")
        print(f"Edges > 0.1: {int((recon_adj > 0.1).sum().item())}")
        
        # Compare input vs reconstructed
        diff = (recon_adj - adj).abs()
        print(f"\n|Input - Recon|: mean={diff.mean():.4f}, max={diff.max():.4f}")
    
    print("\n" + "="*60)
    print("Reconstruction to ZX Graph")
    print("="*60)
    
    recon_adj_cpu = recon_adj[0].cpu()
    node_feat_cpu = node_features[0].cpu()
    
    # Count active nodes
    active_count = (node_feat_cpu.sum(dim=1) > 0.1).sum().item()
    print(f"Active nodes (features sum > 0.1): {active_count}")
    
    # Try different thresholds
    for threshold in [0.5, 0.3, 0.1]:
        print(f"\n--- Threshold: {threshold} ---")
        graph, error = adj_to_zx_graph(recon_adj_cpu, node_feat_cpu, threshold=threshold)
        
        if graph is None:
            print(f"  Graph construction failed: {error}")
            continue
        
        print(f"  ZX graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
        
        # Count vertex types
        vtype_counts = {'Z': 0, 'X': 0, 'B': 0, 'H': 0}
        for v in graph.vertices():
            vt = graph.type(v)
            if vt == zx.VertexType.Z:
                vtype_counts['Z'] += 1
            elif vt == zx.VertexType.X:
                vtype_counts['X'] += 1
            elif vt == zx.VertexType.BOUNDARY:
                vtype_counts['B'] += 1
            elif vt == zx.VertexType.H_BOX:
                vtype_counts['H'] += 1
        print(f"  Vertex types: {vtype_counts}")
        
        # Check boundary degrees
        boundary_degrees = []
        for v in graph.vertices():
            if graph.type(v) == zx.VertexType.BOUNDARY:
                boundary_degrees.append(graph.vertex_degree(v))
        print(f"  Boundary degrees: {boundary_degrees}")
        if boundary_degrees:
            all_degree_1 = all(d == 1 for d in boundary_degrees)
            print(f"  All boundaries degree 1: {all_degree_1}")
        
        # Try extraction
        try:
            circuit = zx.extract_circuit(graph.copy())
            print(f"  EXTRACTION SUCCESS: {circuit.qubits} qubits, {len(circuit.gates)} gates")
        except Exception as e:
            print(f"  EXTRACTION FAILED: {type(e).__name__}")
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Node Features Analysis")
    print("="*60)
    
    print("\nFirst 10 active nodes (node_features):")
    for i in range(min(10, node_feat_cpu.shape[0])):
        feat = node_feat_cpu[i]
        if feat.sum() > 0.1:
            print(f"  Node {i}: {feat.tolist()}")


if __name__ == "__main__":
    main()
