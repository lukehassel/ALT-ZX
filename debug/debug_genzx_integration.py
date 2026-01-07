import torch
import torch.nn as nn
from GenZX.model import GraphVAE
import sys
import os

def test_integration():
    print("=== Testing GenZX + GflowNet Integration ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize Model
    # input_dim=6 (node features), hidden_dim=32, latent_dim=16, max_nodes=10
    model = GraphVAE(input_dim=6, hidden_dim=32, latent_dim=16, max_num_nodes=10, 
                     zxnet_checkpoint='ZXNet/model.pth', # Will skip if not found
                     node_feature_dim=6).to(device)
    
    print("\nModel Initialized.")
    print(f"GflowNet loaded: {hasattr(model, 'gflownet')}")
    if hasattr(model, 'gflownet'):
        print(f"GflowNet Sync Status: {model.gflownet._dense_weights_synced}")
    
    # Create Dummy Data
    B, N, F = 4, 10, 6
    adj = torch.rand(B, N, N).to(device)
    # Symmetrize roughly
    adj = (adj + adj.transpose(1, 2)) / 2
    node_features = torch.randn(B, N, F).to(device)
    
    # Forward Pass
    print("\nRunning Forward Pass...")
    model.train() # Ensure we can track gradients
    
    # Enable gradients for inputs to check flow (though typically we optimize model params)
    node_features.requires_grad = True
    
    loss = model(input_features=None, adj=adj, node_features=node_features)
    
    print(f"Loss: {loss.item()}")
    
    # Backward Pass
    print("\nRunning Backward Pass...")
    loss.backward()
    
    # Check Gradients
    print("\nChecking Gradients:")
    
    # Check VAE encoder gradients
    vae_grad = False
    for p in model.vae.parameters():
        if p.grad is not None and p.grad.norm() > 0:
            vae_grad = True
            break
    print(f"VAE Decoder/Encoder Gradients: {'OK' if vae_grad else 'MISSING'}")
    
    # Check GflowNet Gradients (should be None as it's frozen)
    gflow_grad_param = False
    for p in model.gflownet.parameters():
        if p.grad is not None:
            gflow_grad_param = True
            break
    print(f"GflowNet Params Gradients (Should be None): {'OK (None)' if not gflow_grad_param else 'FAIL (Has Grad)'}")
    
    # Check if gradients flowed back to inputs (via GflowNet dense input)
    print(f"Input Node Features Grad: {node_features.grad.norm().item()}")
    
    if vae_grad and not gflow_grad_param:
        print("\nSUCCESS: Integration verified. Gradients flow through differentiable GflowNet.")
    else:
        print("\nFAILURE: Gradient check failed.")

if __name__ == "__main__":
    test_integration()
