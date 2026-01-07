import torch
import torch.nn.functional as F
import pyzx as zx
import os
import sys
import random
from tqdm import tqdm
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZX.model import GraphVAE
from GenZX.data_loader import GraphVAEDataset
from zx_loader import pyzx_graph_to_pyg, reconstruct_pyzx_from_6feat
from GenZX.dataset import create_random_circuit_graph
from mpo.fidelity import get_fidelity
from ZXNet.model import ZXNet

CONFIG = {
    'model_path': 'GenZX/checkpoints/model.pth',
    'num_samples': 1000,
    'noise_scale': 0,
    'max_num_nodes': 64,
    'edge_threshold': 0.1,
    'num_input_graphs': 100,
}


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


def adj_to_zx_graph(adj_matrix, node_features, threshold=0.5):
    from torch_geometric.data import Data
    
    if adj_matrix.is_cuda:
        adj_matrix = adj_matrix.cpu()
    if node_features.is_cuda:
        node_features = node_features.cpu()
    
    n_nodes = adj_matrix.shape[0]
    feature_mask = node_features.sum(dim=1) > 0.1
    binary_adj = (adj_matrix > threshold).float()
    degree = binary_adj.sum(dim=1)
    connectivity_mask = degree > 0
    active_mask = feature_mask & connectivity_mask
    active_indices = torch.where(active_mask)[0].tolist()
    
    if not active_indices:
        return None
    
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
        return None
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    data = Data(x=active_node_features, edge_index=edge_index, edge_attr=edge_attr)
    return reconstruct_pyzx_from_6feat(data)


def pad_circuit_to_qubits(circuit, target_qubits):
    if circuit.num_qubits >= target_qubits:
        return circuit
    
    padded = QuantumCircuit(target_qubits)
    
    for instruction, qargs, cargs in circuit.data:
        new_qargs = [padded.qubits[q._index] for q in qargs]
        padded.append(instruction, new_qargs, cargs)
    
    return padded


def sample_and_check(model, input_data, input_zx_graph, input_pyg, zxnet_model, noise_scale=0.1, edge_threshold=0.1, max_nodes=64):
    model.eval()
    
    with torch.no_grad():
        node_features = input_data['node_features'].unsqueeze(0).float().cuda()
        adj = input_data['adj'].unsqueeze(0).float().cuda()
        
        z_mu, z_lsgms = model.vae.encode(node_features, adj)
        noise = torch.randn_like(z_mu) * noise_scale
        z_perturbed = z_mu + noise
        
        decoded = model.vae.decode(z_perturbed)
        out = torch.sigmoid(decoded)
        recon_adj = model.recover_adj_differentiable_batch(out)
        
        fixed_adj = fix_boundary_degrees(recon_adj[0].cpu(), node_features[0].cpu(), threshold=0.05)
        recon_zx = adj_to_zx_graph(fixed_adj, node_features[0].cpu(), threshold=edge_threshold)
        
        if recon_zx is None:
            return False, "no_graph", None, None
        
        zxnet_score = None
        try:
            recon_pyg = pyzx_graph_to_pyg(recon_zx)
            if recon_pyg is not None and input_pyg is not None:
                def pad_to_dense(pyg_data, max_n):
                    n = pyg_data.x.shape[0]
                    x_padded = torch.zeros(max_n, pyg_data.x.shape[1])
                    x_padded[:n] = pyg_data.x
                    adj_padded = torch.zeros(max_n, max_n)
                    ei = pyg_data.edge_index
                    adj_padded[ei[0], ei[1]] = 1.0
                    mask = torch.zeros(max_n, dtype=torch.bool)
                    mask[:n] = True
                    return x_padded, adj_padded, mask
                
                x1, adj1, mask1 = pad_to_dense(input_pyg, max_nodes)
                x2, adj2, mask2 = pad_to_dense(recon_pyg, max_nodes)
                
                x1, adj1, mask1 = x1.unsqueeze(0).cuda(), adj1.unsqueeze(0).cuda(), mask1.unsqueeze(0).cuda()
                x2, adj2, mask2 = x2.unsqueeze(0).cuda(), adj2.unsqueeze(0).cuda(), mask2.unsqueeze(0).cuda()
                
                emb1 = zxnet_model.forward_one_graph_dense(x1, adj1, mask1)
                emb2 = zxnet_model.forward_one_graph_dense(x2, adj2, mask2)
                
                combined = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
                out_zx = zxnet_model.fc1(combined)
                out_zx = F.relu(out_zx)
                out_zx = zxnet_model.fc2(out_zx)
                zxnet_score = torch.sigmoid(out_zx).item()
        except Exception:
            pass
        
        try:
            recon_circuit = zx.extract_circuit(recon_zx.copy())
            recon_qasm = recon_circuit.to_qasm()
            recon_qc = QuantumCircuit.from_qasm_str(recon_qasm)
            
            try:
                input_circuit = zx.extract_circuit(input_zx_graph.copy())
                input_qasm = input_circuit.to_qasm()
                input_qc = QuantumCircuit.from_qasm_str(input_qasm)
                
                target_qubits = max(input_qc.num_qubits, recon_qc.num_qubits)
                input_qc_padded = pad_circuit_to_qubits(input_qc, target_qubits)
                recon_qc_padded = pad_circuit_to_qubits(recon_qc, target_qubits)
                
                qubit_mismatch = input_qc.num_qubits != recon_qc.num_qubits
                
                try:
                    result = get_fidelity(input_qc_padded, recon_qc_padded)
                    fidelity = result['fidelity']
                    if qubit_mismatch:
                        return True, f"success_padded_{input_qc.num_qubits}_vs_{recon_qc.num_qubits}", fidelity, zxnet_score
                    return True, "success", fidelity, zxnet_score
                except Exception as e:
                    return True, f"success_fidelity_error_{type(e).__name__}", None, zxnet_score
            except Exception as e:
                return True, "success_input_extract_failed", None, zxnet_score
                
        except Exception as e:
            return False, str(type(e).__name__), None, zxnet_score


def main():
    cfg = CONFIG
    random.seed(42)
    torch.manual_seed(42)
    
    print(f"=== GenZX Extractability + Fidelity Evaluation ===")
    print(f"Samples: {cfg['num_samples']}")
    print(f"Model: {cfg['model_path']}")
    print(f"Noise scale: {cfg['noise_scale']}")
    
    print(f"\nLoading model...")
    model = GraphVAE(
        input_dim=cfg['max_num_nodes'],
        hidden_dim=64,
        latent_dim=1024,
        max_num_nodes=cfg['max_num_nodes'],
        pool='sum',
        lambda_zxnet=1.0
    ).cuda()
    
    checkpoint = torch.load(cfg['model_path'], map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded!")
    
    print(f"\nLoading ZXNet...")
    zxnet = ZXNet(num_node_features=6).cuda()
    zxnet_ckpt = torch.load('ZXNet/model.pth', map_location='cuda', weights_only=False)
    if isinstance(zxnet_ckpt, dict) and 'model_state_dict' in zxnet_ckpt:
        zxnet.load_state_dict(zxnet_ckpt['model_state_dict'])
    else:
        zxnet.load_state_dict(zxnet_ckpt)
    zxnet.eval()
    print("ZXNet loaded!")
    
    print(f"\nGenerating {cfg['num_input_graphs']} input graphs...")
    input_data_list = []
    for _ in range(cfg['num_input_graphs']):
        num_qubits = random.randint(3, 6)
        depth = random.randint(3, 6)
        graph = create_random_circuit_graph(num_qubits, depth, seed=random.randint(0, 1000000))
        if graph is not None and graph.num_vertices() < cfg['max_num_nodes']:
            pyg_data = pyzx_graph_to_pyg(graph)
            if pyg_data is not None:
                input_data_list.append((pyg_data, graph))
    
    print(f"Generated {len(input_data_list)} valid input graphs")
    
    datasets_with_graphs = []
    for pyg_data, zx_graph in input_data_list:
        dataset = GraphVAEDataset([pyg_data], max_nodes=cfg['max_num_nodes'])
        datasets_with_graphs.append((dataset, zx_graph, pyg_data))
    
    print(f"\nSampling {cfg['num_samples']} times...")
    success_count = 0
    failure_reasons = {}
    success_reasons = {}
    fidelities = []
    zxnet_scores = []
    
    samples_per_graph = cfg['num_samples'] // len(datasets_with_graphs)
    
    for dataset, zx_graph, pyg_data in tqdm(datasets_with_graphs, desc="Processing graphs"):
        input_data = dataset[0]
        for _ in range(samples_per_graph):
            success, reason, fidelity, zxnet_score = sample_and_check(
                model, input_data, zx_graph, pyg_data, zxnet,
                noise_scale=cfg['noise_scale'],
                edge_threshold=cfg['edge_threshold'],
                max_nodes=cfg['max_num_nodes']
            )
            if success:
                success_count += 1
                success_reasons[reason] = success_reasons.get(reason, 0) + 1
                if fidelity is not None:
                    fidelities.append(fidelity)
                if zxnet_score is not None:
                    zxnet_scores.append(zxnet_score)
                zxnet_str = f", ZXNet: {zxnet_score:.4f}" if zxnet_score else ""
                fid_str = f"Fidelity: {fidelity:.4f}" if fidelity else "Fidelity: N/A"
                print(f"  ✓ Extracted! {fid_str}{zxnet_str}")
            else:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    total = len(datasets_with_graphs) * samples_per_graph
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Successful extractions: {success_count} ({100*success_count/total:.2f}%)")
    
    if fidelities:
        import numpy as np
        fidelities_arr = np.array(fidelities)
        print(f"\nFidelity Statistics ({len(fidelities)} samples with fidelity):")
        print(f"  Mean:   {fidelities_arr.mean():.4f}")
        print(f"  Std:    {fidelities_arr.std():.4f}")
        print(f"  Min:    {fidelities_arr.min():.4f}")
        print(f"  Max:    {fidelities_arr.max():.4f}")
        print(f"  Median: {np.median(fidelities_arr):.4f}")
    
    if zxnet_scores:
        import numpy as np
        zxnet_arr = np.array(zxnet_scores)
        print(f"\nZXNet Similarity Statistics ({len(zxnet_scores)} samples with ZXNet score):")
        print(f"  Mean:   {zxnet_arr.mean():.4f}")
        print(f"  Std:    {zxnet_arr.std():.4f}")
        print(f"  Min:    {zxnet_arr.min():.4f}")
        print(f"  Max:    {zxnet_arr.max():.4f}")
        print(f"  Median: {np.median(zxnet_arr):.4f}")
    
    print(f"\nSuccess breakdown:")
    for reason, count in sorted(success_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print(f"\nFailure breakdown:")
    for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()
