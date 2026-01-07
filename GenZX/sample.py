import torch
import pyzx as zx
import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, EfficientSU2

from GenZX.model import GraphVAE
from GenZX.data_loader import GraphVAEDataset
from mpo.fidelity import get_fidelity
from circuit_utils import create_random_circuit_with_universal_gates


from zx_loader import pyzx_graph_to_pyg

CONFIG = {
    'model_path': 'GenZX/checkpoints/model.pth',
    'num_qubits': 5,
    'depth': 10,
    'num_samples': 5,
    'noise_scale': 0.01,
    'seed': 42,
    'max_num_nodes': 64,
    'edge_threshold': 0.1,
}


def circuit_to_zx_graph(qc):
    from qiskit import qasm2
    qasm_str = qasm2.dumps(qc)
    circuit = zx.Circuit.from_qasm(qasm_str)
    graph = circuit.to_graph()
    return graph, circuit


def fix_boundary_degrees(adj_matrix, node_features, threshold=0.1):
    adj = adj_matrix.clone()
    n_nodes = adj.shape[0]
    
    boundary_indices = []
    for i in range(n_nodes):
        feat = node_features[i]
        if feat.sum() > 0.1:
            node_type = feat[1].item()
            if abs(node_type - 0.0) < 0.1:
                boundary_indices.append(i)
    
    for bidx in boundary_indices:
        row = adj[bidx, :].clone()
        col = adj[:, bidx].clone()
        
        adj[bidx, :] = 0
        adj[:, bidx] = 0
        
        row[bidx] = 0
        col[bidx] = 0
        
        combined = torch.max(row, col)
        
        if combined.max() > threshold:
            best_neighbor = combined.argmax().item()
            adj[bidx, best_neighbor] = 1.0
            adj[best_neighbor, bidx] = 1.0
    
    return adj


def adj_to_zx_graph(adj_matrix, node_features, threshold=0.5, use_connectivity=True):
    from torch_geometric.data import Data
    from zx_loader import reconstruct_pyzx_from_6feat
    
    if adj_matrix.is_cuda:
        adj_matrix = adj_matrix.cpu()
    if node_features.is_cuda:
        node_features = node_features.cpu()
    
    n_nodes = adj_matrix.shape[0]
    
    feature_mask = node_features.sum(dim=1) > 0.1
    
    if use_connectivity:
        binary_adj = (adj_matrix > threshold).float()
        degree = binary_adj.sum(dim=1)
        connectivity_mask = degree > 0
        active_mask = feature_mask & connectivity_mask
    else:
        active_mask = feature_mask
    
    active_indices = torch.where(active_mask)[0].tolist()
    
    if not active_indices:
        return None
    
    old_to_new = {old: new for new, old in enumerate(active_indices)}
    
    active_node_features = node_features[active_indices]
    
    src_list = []
    dst_list = []
    edge_attr_list = []
    
    for i, old_i in enumerate(active_indices):
        for j, old_j in enumerate(active_indices):
            if i >= j:
                continue
            if adj_matrix[old_i, old_j] > threshold:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                edge_attr_list.extend([[0.0], [0.0]])
    
    if not src_list:
        return None
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    data = Data(
        x=active_node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return reconstruct_pyzx_from_6feat(data)


def sample_model(model, input_data, num_samples=1, noise_scale=0.1, verbose=True):
    model.eval()
    samples = []
    
    with torch.no_grad():
        node_features = input_data['node_features'].unsqueeze(0).float().cuda()
        adj = input_data['adj'].unsqueeze(0).float().cuda()
        
        if verbose:
            print("\n" + "="*60)
            print("RAW MODEL OUTPUT ANALYSIS")
            print("="*60)
            print(f"\n[INPUT]")
            print(f"   Node features shape: {node_features.shape}")
            print(f"   Node features range: [{node_features.min():.2f}, {node_features.max():.2f}]")
            print(f"   Adjacency shape: {adj.shape}")
        
        z_mu, z_lsgms = model.vae.encode(node_features, adj)
        
        if verbose:
            print(f"\n[ENCODER OUTPUT]")
            print(f"   z_mu shape: {z_mu.shape}")
            print(f"   z_mu range: [{z_mu.min():.4f}, {z_mu.max():.4f}]")
            print(f"   z_mu mean: {z_mu.mean():.4f}, std: {z_mu.std():.4f}")
            print(f"   z_lsgms (log-variance) range: [{z_lsgms.min():.4f}, {z_lsgms.max():.4f}]")
        
        for i in range(num_samples):
            noise = torch.randn_like(z_mu) * noise_scale
            z_perturbed = z_mu + noise
            
            decoded = model.vae.decode(z_perturbed)
            out = torch.sigmoid(decoded)
            
            if verbose and i == 0:
                print(f"\n[DECODER OUTPUT - Sample 1]")
                print(f"   Raw decoded shape: {decoded.shape}")
                print(f"   Raw decoded range: [{decoded.min():.4f}, {decoded.max():.4f}]")
                print(f"   After sigmoid (out) range: [{out.min():.4f}, {out.max():.4f}]")
                print(f"   After sigmoid mean: {out.mean():.4f}")
                
                values = out.flatten().cpu().numpy()
                bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                print(f"\n   Value distribution (histogram):")
                for j in range(len(bins)-1):
                    count = ((values >= bins[j]) & (values < bins[j+1])).sum()
                    pct = count / len(values) * 100
                    print(f"     [{bins[j]:.1f}-{bins[j+1]:.1f}]: {pct:5.1f}%")
            
            recon_adj = model.recover_adj_differentiable_batch(out)
            
            if verbose and i == 0:
                print(f"\n[ADJACENCY MATRIX - Sample 1]")
                print(f"   Shape: {recon_adj.shape}")
                print(f"   Range: [{recon_adj.min():.4f}, {recon_adj.max():.4f}]")
                print(f"   Mean: {recon_adj.mean():.4f}")
                print(f"   Is symmetric (diff): {(recon_adj - recon_adj.transpose(-1,-2)).abs().max():.6f}")
                
                corner = recon_adj[0, :8, :8].cpu().numpy()
                print(f"\n   Upper-left 8x8 corner:")
                print("        " + "  ".join([f"{i:5d}" for i in range(8)]))
                for row in range(8):
                    print(f"     {row}: " + "  ".join([f"{corner[row, col]:.3f}" for col in range(8)]))
            
            samples.append({
                'adj': recon_adj[0].cpu(),
                'node_features': node_features[0].cpu(),
                'z': z_perturbed[0].cpu()
            })
    
    return samples


def main():
    cfg = CONFIG
    
    print(f"=== GenZX Model Sampling ===")
    print(f"Random seed: {cfg['seed']}")
    
    print(f"\n1. Generating random {cfg['num_qubits']}-qubit circuit with depth {cfg['depth']}...")
    qc = create_random_circuit_with_universal_gates(cfg['num_qubits'], cfg['depth'], seed=cfg['seed'])
    print(f"   Circuit: {qc.num_qubits} qubits, {qc.depth()} depth, {len(qc.data)} gates")
    
    print("\n2. Converting to ZX-graph...")
    zx_graph, zx_circuit = circuit_to_zx_graph(qc)
    print(f"   ZX-graph: {zx_graph.num_vertices()} vertices, {zx_graph.num_edges()} edges")
    
    print("\n3. Converting to PyG format...")
    pyg_data = pyzx_graph_to_pyg(zx_graph, max_nodes=None)
    
    if pyg_data.num_nodes > 256:
        print(f"   WARNING: Graph has {pyg_data.num_nodes} nodes, which exceeds the supported limit of 256.")
    
    print(f"   PyG data: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")
    
    print(f"\n4. Loading model from {cfg['model_path']}...")
    model = GraphVAE(
        input_dim=cfg['max_num_nodes'],
        hidden_dim=64,
        latent_dim=1024,
        max_num_nodes=cfg['max_num_nodes'],
        pool='sum',
        lambda_zxnet=1.0
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    try:
        checkpoint = torch.load(cfg['model_path'], map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   Model loaded successfully!")
    except FileNotFoundError:
        print(f"   WARNING: Model checkpoint not found at {cfg['model_path']}")
        print("   Using randomly initialized model for demo purposes.")
    
    dataset = GraphVAEDataset([pyg_data], max_nodes=cfg['max_num_nodes'])
    input_data = dataset[0]
    
    print(f"\n5. Sampling {cfg['num_samples']} graphs from model (noise={cfg['noise_scale']})...")
    samples = sample_model(model, input_data, num_samples=cfg['num_samples'], noise_scale=cfg['noise_scale'])
    
    print("\n6. Analyzing reconstructed graphs...")
    
    from GflowEncoder.model import GraphEncoder
    gflow_encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64).cuda()
    gflow_encoder.load_state_dict(torch.load('GflowEncoder/encoder.pth', map_location='cuda', weights_only=True))
    gflow_encoder.eval()
    gflow_centroid = torch.load('GflowEncoder/valid_centroid.pt', map_location='cuda', weights_only=True)
    print("   Loaded GflowEncoder for embedding-based validity scoring")
    
    for i, sample in enumerate(samples):
        recon_adj = sample['adj']
        node_feat = sample['node_features']
        print(f"\n{'='*60}")
        print(f"   Sample {i+1} Analysis:")
        print(f"{'='*60}")
        
        print("\n   [GflowEncoder Embedding Score]")
        with torch.no_grad():
            recon_adj_gpu = recon_adj.unsqueeze(0).cuda()
            node_feat_gpu = node_feat.unsqueeze(0).cuda()
            embedding = gflow_encoder.forward_dense(node_feat_gpu, recon_adj_gpu)
            similarity = (embedding @ gflow_centroid.unsqueeze(1)).squeeze()
            score = (similarity + 1) / 2
            print(f"      Validity score: {score.item():.4f}")
            if score.item() > 0.7:
                print(f"      GflowEncoder says: LIKELY VALID ✓")
            elif score.item() > 0.5:
                print(f"      GflowEncoder says: POSSIBLY VALID ~")
            else:
                print(f"      GflowEncoder says: LIKELY INVALID ✗")
        
        print("\n   [Adjacency Matrix Stats]")
        print(f"      Shape: {recon_adj.shape}")
        print(f"      Value range: [{recon_adj.min():.4f}, {recon_adj.max():.4f}]")
        print(f"      Mean: {recon_adj.mean():.4f}")
        
        binary_adj = (recon_adj > cfg['edge_threshold']).float()
        num_edges_binary = int(binary_adj.sum().item() / 2)
        print(f"      Edges (threshold={cfg['edge_threshold']}): {num_edges_binary}")
        
        density = binary_adj.sum().item() / (binary_adj.shape[0] * binary_adj.shape[1])
        print(f"      Density: {density:.4f}")
        
        print("\n   [Node Features Stats]")
        print(f"      Shape: {node_feat.shape}")
        
        type_counts = {}
        active_nodes = 0
        for j in range(node_feat.shape[0]):
            feat = node_feat[j]
            if feat.sum() > 0.1:
                active_nodes += 1
                type_idx = feat[:4].argmax().item()
                types = ['Z-spider', 'X-spider', 'Boundary', 'H-box']
                type_name = types[type_idx]
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        print(f"      Active nodes: {active_nodes} / {node_feat.shape[0]}")
        print(f"      Node type breakdown:")
        for t, c in type_counts.items():
            print(f"         {t}: {c}")
        
        print("\n   [ZX Graph Reconstruction]")
        
        fixed_adj = fix_boundary_degrees(recon_adj, node_feat, threshold=0.05)
        print("      Applied boundary degree fix (keeping highest prob edge per boundary)")
        
        recon_zx_before = adj_to_zx_graph(recon_adj, node_feat, threshold=cfg['edge_threshold'])
        recon_zx = adj_to_zx_graph(fixed_adj, node_feat, threshold=cfg['edge_threshold'])
        
        os.makedirs('images', exist_ok=True)
        try:
            if recon_zx_before is not None:
                zx.draw(recon_zx_before, labels=False)
                import matplotlib.pyplot as plt
                plt.savefig(f'images/sample_{i+1}_before.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      Saved: images/sample_{i+1}_before.png")
            if recon_zx is not None:
                zx.draw(recon_zx, labels=False)
                plt.savefig(f'images/sample_{i+1}_after.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      Saved: images/sample_{i+1}_after.png")
        except Exception as e:
            print(f"      Image save failed: {e}")
        
        if recon_zx is None:
            print("      FAILED: No active nodes found")
            continue
        
        print(f"      Vertices: {recon_zx.num_vertices()}")
        print(f"      Edges: {recon_zx.num_edges()}")
        
        zx_types = {'Z': 0, 'X': 0, 'B': 0, 'H': 0, 'Other': 0}
        for v in recon_zx.vertices():
            vtype = recon_zx.type(v)
            if vtype == zx.VertexType.Z:
                zx_types['Z'] += 1
            elif vtype == zx.VertexType.X:
                zx_types['X'] += 1
            elif vtype == zx.VertexType.BOUNDARY:
                zx_types['B'] += 1
            elif vtype == zx.VertexType.H_BOX:
                zx_types['H'] += 1
            else:
                zx_types['Other'] += 1
        
        print(f"      ZX vertex types: Z={zx_types['Z']}, X={zx_types['X']}, B={zx_types['B']}, H={zx_types['H']}")
        
        print("\n   [GflowEncoder on Reconstructed PyZX Graph]")
        try:
            recon_pyg = pyzx_graph_to_pyg(recon_zx)
            if recon_pyg is not None and recon_pyg.edge_index.shape[1] > 0:
                with torch.no_grad():
                    recon_x = recon_pyg.x.unsqueeze(0).cuda()
                    n = recon_x.shape[1]
                    recon_adj_dense = torch.zeros(1, n, n, device='cuda')
                    ei = recon_pyg.edge_index
                    recon_adj_dense[0, ei[0], ei[1]] = 1.0
                    
                    if n < 256:
                        pad_x = torch.zeros(1, 256, 6, device='cuda')
                        pad_x[0, :n, :] = recon_x
                        pad_adj = torch.zeros(1, 256, 256, device='cuda')
                        pad_adj[0, :n, :n] = recon_adj_dense[0]
                        recon_x = pad_x
                        recon_adj_dense = pad_adj
                    
                    emb = gflow_encoder.forward_dense(recon_x, recon_adj_dense)
                    sim = (emb @ gflow_centroid.unsqueeze(1)).squeeze()
                    gflow_score = (sim + 1) / 2
                    print(f"      Reconstructed graph score: {gflow_score.item():.4f}")
                    if gflow_score.item() > 0.7:
                        print("      GflowEncoder says: LIKELY HAS GFLOW ✓")
                    elif gflow_score.item() > 0.5:
                        print("      GflowEncoder says: POSSIBLY HAS GFLOW ~")
                    else:
                        print("      GflowEncoder says: LIKELY NO GFLOW ✗")
            else:
                print("      Could not convert to PyG format")
        except Exception as e:
            print(f"      Error checking gflow: {e}")
        
        print("\n   [Validity Checks]")
        has_boundaries = zx_types['B'] > 0
        print(f"      Has boundary nodes: {has_boundaries} (required for circuit extraction)")
        
        boundary_degrees = []
        for v in recon_zx.vertices():
            if recon_zx.type(v) == zx.VertexType.BOUNDARY:
                boundary_degrees.append(recon_zx.vertex_degree(v))
        if boundary_degrees:
            print(f"      Boundary degrees: {boundary_degrees}")
            valid_boundaries = all(d == 1 for d in boundary_degrees)
            print(f"      All boundaries have degree 1: {valid_boundaries} (REQUIRED for circuit!)")
        
        degrees = [recon_zx.vertex_degree(v) for v in recon_zx.vertices()]
        print(f"      Degree range: [{min(degrees)}, {max(degrees)}], avg: {sum(degrees)/len(degrees):.1f}")
        
        print("\n   [Circuit Extraction]")
        try:
            recon_circuit = zx.extract_circuit(recon_zx.copy())
            print(f"      SUCCESS! Circuit extracted")
            recon_qc = QuantumCircuit.from_qasm_str(recon_circuit.to_qasm())
            print(f"      Qubits: {recon_qc.num_qubits}, Gates: {len(recon_qc.data)}")
            
            if recon_qc.num_qubits == qc.num_qubits:
                result = get_fidelity(qc, recon_qc)
                print(f"      Fidelity with input: {result['fidelity']:.6f}")
            else:
                print(f"      Qubit mismatch: input={qc.num_qubits}, output={recon_qc.num_qubits}")
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {e}")
    
    print(f"\n{'='*60}")
    print("=== Done ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
