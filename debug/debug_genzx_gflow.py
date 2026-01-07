"""
Thorough debugging of GflowNet-GenZX integration.

Checks:
1. Gradient flow from loss_gflow back to VAE parameters
2. Individual loss component magnitudes
3. GflowNet weight sync (sparse vs dense layers)
4. GflowNet forward_dense behavior on real data
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GenZX.model import GraphVAE
from GenZX.data_loader import GraphVAEDataset, load_dataset
from GflowNet.model import GflowNet


def debug_weight_sync():
    """Check if GflowNet sparse and dense weights are synchronized."""
    print("\n" + "="*60)
    print("1. GFLOWNET WEIGHT SYNCHRONIZATION CHECK")
    print("="*60)
    
    gflownet = GflowNet(num_node_features=6)
    
    # Load checkpoint
    ckpt_path = 'GflowNet/model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            gflownet.load_state_dict(ckpt['model_state_dict'])
        else:
            gflownet.load_state_dict(ckpt)
        print(f"   Loaded checkpoint from {ckpt_path}")
    else:
        print(f"   WARNING: No checkpoint found at {ckpt_path}")
        return
    
    # Check sync status
    print(f"\n   Dense weights synced: {gflownet._dense_weights_synced}")
    
    # Force sync and compare weights
    gflownet._sync_dense_weights()
    
    # Compare conv1 sparse vs dense
    sparse_weight = gflownet.conv1.lin.weight.data
    dense_weight = gflownet.conv1_dense.lin.weight.data
    weight_diff = (sparse_weight - dense_weight).abs().max().item()
    print(f"\n   Layer conv1 weight diff (sparse vs dense): {weight_diff:.6f}")
    
    sparse_bn = gflownet.bn1.weight.data
    dense_bn = gflownet.bn1_dense.weight.data
    bn_diff = (sparse_bn - dense_bn).abs().max().item()
    print(f"   Layer bn1 weight diff: {bn_diff:.6f}")
    
    if weight_diff < 1e-6 and bn_diff < 1e-6:
        print("\n   ✓ Weights are synchronized correctly!")
    else:
        print("\n   ✗ WARNING: Weight mismatch detected!")


def debug_forward_dense():
    """Test GflowNet forward_dense on synthetic data."""
    print("\n" + "="*60)
    print("2. GFLOWNET FORWARD_DENSE BEHAVIOR")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gflownet = GflowNet(num_node_features=6).to(device)
    ckpt = torch.load('GflowNet/model.pth', map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        gflownet.load_state_dict(ckpt['model_state_dict'])
    gflownet.eval()
    
    # Create test inputs
    B, N, F = 4, 64, 6
    
    # Random adjacency (soft probabilities)
    adj_random = torch.rand(B, N, N, device=device)
    adj_random = (adj_random + adj_random.transpose(-1, -2)) / 2  # Symmetrize
    
    # Identity-like adjacency (strong diagonal)
    adj_identity = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    
    # Dense adjacency (all ones)
    adj_dense = torch.ones(B, N, N, device=device)
    
    # Random node features
    node_feat = torch.randn(B, N, F, device=device)
    
    print(f"\n   Testing on batch_size={B}, num_nodes={N}")
    
    with torch.no_grad():
        prob_random = gflownet.forward_dense(node_feat, adj_random)
        prob_identity = gflownet.forward_dense(node_feat, adj_identity)
        prob_dense = gflownet.forward_dense(node_feat, adj_dense)
    
    print(f"\n   Random adjacency -> P(gflow): {prob_random.mean():.4f} (range: {prob_random.min():.4f}-{prob_random.max():.4f})")
    print(f"   Identity adjacency -> P(gflow): {prob_identity.mean():.4f}")
    print(f"   Dense adjacency -> P(gflow): {prob_dense.mean():.4f}")
    
    # Test gradient flow
    print("\n   Testing gradient flow through forward_dense...")
    adj_test = torch.rand(1, N, N, device=device, requires_grad=True)
    adj_test_sym = (adj_test + adj_test.transpose(-1, -2)) / 2
    node_test = torch.randn(1, N, F, device=device, requires_grad=True)
    
    gflownet.train()  # Enable grad tracking in BN
    prob = gflownet.forward_dense(node_test, adj_test_sym)
    loss = -torch.log(prob + 1e-6)
    loss.backward()
    
    if adj_test.grad is not None and adj_test.grad.abs().sum() > 0:
        print(f"   ✓ Gradients flow to adjacency input: grad norm = {adj_test.grad.norm():.4f}")
    else:
        print("   ✗ No gradients to adjacency input!")
    
    if node_test.grad is not None and node_test.grad.abs().sum() > 0:
        print(f"   ✓ Gradients flow to node features: grad norm = {node_test.grad.norm():.4f}")
    else:
        print("   ✗ No gradients to node features!")


def debug_loss_components():
    """Check individual loss component magnitudes during a forward pass."""
    print("\n" + "="*60)
    print("3. LOSS COMPONENT ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load a small batch of real data
    raw_data = load_dataset('combined/dataset_6feat.pt')
    graphs = []
    for sample in raw_data[:100]:
        g1, g2 = sample[0], sample[1]
        if hasattr(g1, 'x') and g1.x.shape[0] <= 64:
            graphs.append(g1)
            break
    
    if not graphs:
        print("   ERROR: No suitable graphs found in dataset")
        return
    
    dataset = GraphVAEDataset(graphs, max_nodes=64)
    
    # Create model
    model = GraphVAE(
        input_dim=64,
        hidden_dim=64,
        latent_dim=256,
        max_num_nodes=64,
        pool='sum',
        lambda_zxnet=1.0
    ).to(device)
    
    # Get a batch
    data = dataset[0]
    features = data['features'].unsqueeze(0).float().to(device)
    adj = data['adj'].unsqueeze(0).float().to(device)
    node_features = data['node_features'].unsqueeze(0).float().to(device)
    
    print(f"\n   Input shapes: adj={adj.shape}, node_features={node_features.shape}")
    
    # Manual forward pass to see individual losses
    model.train()
    
    # Encode
    h_decode, z_mu, z_lsgms = model.vae(node_features, adj)
    out = torch.sigmoid(h_decode)
    recon_adj = model.recover_adj_differentiable_batch(out)
    
    print(f"\n   Latent z_mu: mean={z_mu.mean():.4f}, std={z_mu.std():.4f}")
    print(f"   Reconstructed adj: mean={recon_adj.mean():.4f}, range=[{recon_adj.min():.4f}, {recon_adj.max():.4f}]")
    
    # KL Loss
    loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
    loss_kl = loss_kl / (1 * 64 * 64)
    print(f"\n   Loss KL: {loss_kl.item():.6f}")
    
    # ZXNet Loss
    if model.lambda_zxnet > 0:
        loss_zxnet = model.zxnet_loss_batch(adj, recon_adj, node_features)
        print(f"   Loss ZXNet: {loss_zxnet.item():.6f}")
    else:
        loss_zxnet = torch.tensor(0.0)
        print(f"   Loss ZXNet: DISABLED")
    
    # Boundary Loss
    loss_boundary = model.boundary_degree_loss(recon_adj, node_features)
    print(f"   Loss Boundary: {loss_boundary.item():.6f}")
    
    # GflowNet Loss
    gflow_prob = model.gflownet.forward_dense(node_features, recon_adj)
    loss_gflow = -torch.mean(torch.log(gflow_prob + 1e-6))
    print(f"   Loss Gflow: {loss_gflow.item():.6f} (P(gflow)={gflow_prob.mean():.6f})")
    
    # Total
    lambda_boundary = 1.0
    total_loss = (model.lambda_zxnet * loss_zxnet + 
                  loss_kl + 
                  lambda_boundary * loss_boundary +
                  model.lambda_gflow * loss_gflow)
    print(f"\n   Total Loss: {total_loss.item():.6f}")
    print(f"   (lambda_gflow = {model.lambda_gflow})")


def debug_gradient_flow():
    """Check if gradients from loss_gflow reach VAE parameters."""
    print("\n" + "="*60)
    print("4. GRADIENT FLOW FROM GFLOW LOSS TO VAE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    raw_data = load_dataset('combined/dataset_6feat.pt')
    graphs = []
    for sample in raw_data[:100]:
        g1, g2 = sample[0], sample[1]
        if hasattr(g1, 'x') and g1.x.shape[0] <= 64:
            graphs.append(g1)
            break
    
    dataset = GraphVAEDataset(graphs, max_nodes=64)
    
    # Create model
    model = GraphVAE(
        input_dim=64,
        hidden_dim=64,
        latent_dim=256,
        max_num_nodes=64,
        pool='sum',
        lambda_zxnet=0.0  # Disable other losses for isolation
    ).to(device)
    
    # Get batch
    data = dataset[0]
    adj = data['adj'].unsqueeze(0).float().to(device)
    node_features = data['node_features'].unsqueeze(0).float().to(device)
    
    model.train()
    model.zero_grad()
    
    # Forward
    h_decode, z_mu, z_lsgms = model.vae(node_features, adj)
    out = torch.sigmoid(h_decode)
    recon_adj = model.recover_adj_differentiable_batch(out)
    
    # Only GflowNet loss
    gflow_prob = model.gflownet.forward_dense(node_features, recon_adj)
    loss_gflow = -torch.mean(torch.log(gflow_prob + 1e-6))
    
    print(f"\n   Gflow probability: {gflow_prob.item():.6f}")
    print(f"   Gflow loss: {loss_gflow.item():.6f}")
    
    # Backward
    loss_gflow.backward()
    
    # Check gradients on VAE components
    print("\n   Gradient magnitudes on VAE layers:")
    
    # Encoder
    enc_grad = model.vae.conv1.weight.grad
    if enc_grad is not None:
        print(f"      VAE conv1: {enc_grad.norm().item():.6f}")
    else:
        print(f"      VAE conv1: NO GRADIENT")
    
    enc_fc_grad = model.vae.fc_mu.weight.grad
    if enc_fc_grad is not None:
        print(f"      VAE fc_mu: {enc_fc_grad.norm().item():.6f}")
    else:
        print(f"      VAE fc_mu: NO GRADIENT")
    
    # Decoder
    dec_grad = model.vae.fc_decode1.weight.grad
    if dec_grad is not None:
        print(f"      VAE fc_decode1: {dec_grad.norm().item():.6f}")
    else:
        print(f"      VAE fc_decode1: NO GRADIENT")
    
    dec2_grad = model.vae.fc_decode2.weight.grad
    if dec2_grad is not None:
        print(f"      VAE fc_decode2: {dec2_grad.norm().item():.6f}")
    else:
        print(f"      VAE fc_decode2: NO GRADIENT")
    
    # Check GflowNet gradients (should be None since frozen)
    print("\n   GflowNet parameter gradients (should be None/zero):")
    gflow_has_grad = False
    for name, param in model.gflownet.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            gflow_has_grad = True
            print(f"      {name}: {param.grad.norm().item():.6f} ← UNEXPECTED!")
    if not gflow_has_grad:
        print(f"      ✓ All frozen (no gradients)")


def debug_reconstruction_loss():
    """Check if reconstruction loss exists and its magnitude."""
    print("\n" + "="*60)
    print("5. RECONSTRUCTION LOSS ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    raw_data = load_dataset('combined/dataset_6feat.pt')
    graphs = []
    for sample in raw_data[:100]:
        g1, g2 = sample[0], sample[1]
        if hasattr(g1, 'x') and g1.x.shape[0] <= 64:
            graphs.append(g1)
            break
    
    dataset = GraphVAEDataset(graphs, max_nodes=64)
    data = dataset[0]
    adj = data['adj'].unsqueeze(0).float().to(device)
    node_features = data['node_features'].unsqueeze(0).float().to(device)
    
    model = GraphVAE(
        input_dim=64,
        hidden_dim=64,
        latent_dim=256,
        max_num_nodes=64
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        h_decode, z_mu, z_lsgms = model.vae(node_features, adj)
        out = torch.sigmoid(h_decode)
        recon_adj = model.recover_adj_differentiable_batch(out)
    
    # BCE reconstruction loss
    target_upper = adj[0][torch.triu(torch.ones(64, 64, device=device)) == 1]
    pred_upper = out[0]
    
    recon_loss = F.binary_cross_entropy(pred_upper, target_upper)
    print(f"\n   BCE Reconstruction Loss: {recon_loss.item():.6f}")
    
    # Edge statistics
    target_edges = (target_upper > 0.5).sum().item()
    pred_edges = (pred_upper > 0.5).sum().item()
    total_possible = pred_upper.numel()
    
    print(f"   Target edges (>0.5): {target_edges} / {total_possible}")
    print(f"   Predicted edges (>0.5): {pred_edges} / {total_possible}")
    print(f"   Target sparsity: {1 - target_edges/total_possible:.4f}")
    print(f"   Predicted sparsity: {1 - pred_edges/total_possible:.4f}")
    
    # If most predictions are near 0, the model is collapsing to sparse output
    pred_mean = pred_upper.mean().item()
    pred_std = pred_upper.std().item()
    print(f"\n   Prediction mean: {pred_mean:.6f}")
    print(f"   Prediction std: {pred_std:.6f}")
    
    if pred_mean < 0.1 and pred_std < 0.1:
        print("\n   ⚠️  WARNING: Model predictions are collapsed near 0!")
        print("      This means BCE loss encourages 'all zeros' due to sparsity.")
        print("      Consider adding pos_weight to penalize missing edges more.")


def main():
    print("="*60)
    print("GFLOWNET-GENZX INTEGRATION DEBUGGING")
    print("="*60)
    
    debug_weight_sync()
    debug_forward_dense()
    debug_loss_components()
    debug_gradient_flow()
    debug_reconstruction_loss()
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
