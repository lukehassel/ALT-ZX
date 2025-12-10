import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys

# Apply DGL compatibility patch before any DGL imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import dgl_patch
except ImportError:
    pass  # Patch not available, try without it

# --- IMPORTS ---
# Adjust these based on your actual file structure
try:
    from encoder.encoder import FidelityEncoder
    from encoder.dataset import EncoderDataset, collate_quantum_graphs
    from mpo.circuit_utils import get_universal_gate_set
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Ensure 'encoder/encoder.py' and 'encoder/dataset.py' exist.")
    exit(1)

# ==========================================
# 1. Verification Function
# ==========================================
def verify_model(model, loader, device):
    """
    Calculates average cosine similarity for positive and negative pairs.
    """
    print("\n--- Running Numerical Verification ---")
    model.eval()
    
    pos_sims = []
    neg_sims = []
    
    with torch.no_grad():
        # Iterate through a few batches to get stable stats
        for batch_idx, (batch_A, batch_B) in enumerate(loader):
            if batch_idx > 20: break # Check first 20 batches
            
            g1, t1, l1 = batch_A
            g2, t2, l2 = batch_B
            
            g1, t1, l1 = g1.to(device), t1.to(device), l1.to(device)
            g2, t2, l2 = g2.to(device), t2.to(device), l2.to(device)
            
            # Get Embeddings
            z1 = model(g1, t1, l1) # Original Circuits
            z2 = model(g2, t2, l2) # Noisy Equivalent Circuits
            
            # 1. Positive Pair Similarity (Should be High)
            # z1[i] vs z2[i] are equivalent
            pos_sim = F.cosine_similarity(z1, z2)
            pos_sims.append(pos_sim)
            
            # 2. Negative Pair Similarity (Should be Low)
            # Compare z1[i] with z2[i+1] (shifted indices = different circuits)
            z2_shifted = torch.roll(z2, shifts=1, dims=0)
            neg_sim = F.cosine_similarity(z1, z2_shifted)
            neg_sims.append(neg_sim)

    # Aggregate results
    if len(pos_sims) > 0:
        avg_pos = torch.cat(pos_sims).mean().item()
        avg_neg = torch.cat(neg_sims).mean().item()
        
        print(f"Average Similarity (Equivalent Circuits): {avg_pos:.4f} (Goal: > 0.8)")
        print(f"Average Similarity (Random Circuits):     {avg_neg:.4f} (Goal: ~ 0.0)")
        print(f"Discrimination Gap:                       {avg_pos - avg_neg:.4f}")
    else:
        print("No data processed.")

# ==========================================
# 2. Visualization Function
# ==========================================
def visualize_latent_space(model, loader, device, save_name="latent_space_pca.png"):
    """
    Plots the embeddings of Original vs Noisy circuits using PCA.
    Equivalent pairs are connected by lines.
    """
    print(f"\n--- Running Visualization ({save_name}) ---")
    model.eval()
    
    # Take one batch
    try:
        batch_A, batch_B = next(iter(loader))
    except StopIteration:
        print("Loader is empty.")
        return
    
    g1, t1, l1 = batch_A
    g2, t2, l2 = batch_B
    
    with torch.no_grad():
        z1 = model(g1.to(device), t1.to(device), l1.to(device)).cpu().numpy()
        z2 = model(g2.to(device), t2.to(device), l2.to(device)).cpu().numpy()
        
    # We'll plot the first N pairs
    num_pairs_to_plot = min(10, z1.shape[0])
    
    # Combine z1 and z2 for dimensionality reduction to ensure same projection
    combined = np.concatenate([z1[:num_pairs_to_plot], z2[:num_pairs_to_plot]], axis=0)
    
    # Use PCA
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(combined)
    
    # Split back
    r1 = reduced[:num_pairs_to_plot] # Originals
    r2 = reduced[num_pairs_to_plot:] # Noisy equivalents
    
    plt.figure(figsize=(10, 8))
    
    for i in range(num_pairs_to_plot):
        # Plot Original (Circle)
        plt.scatter(r1[i, 0], r1[i, 1], color=f'C{i}', marker='o', s=150, 
                   edgecolor='black', label=f'Pair {i}' if i < 5 else "")
        
        # Plot Noisy Version (X)
        plt.scatter(r2[i, 0], r2[i, 1], color=f'C{i}', marker='X', s=150, 
                   edgecolor='black')
        
        # Draw line connecting them (The "pull" of the contrastive loss)
        plt.plot([r1[i, 0], r2[i, 0]], [r1[i, 1], r2[i, 1]], 
                color=f'C{i}', linestyle='--', alpha=0.6, linewidth=2)

    plt.title(f"Latent Space (PCA)\nConnecting Equivalent Quantum Circuits", fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_name, dpi=150)
    print(f"Saved plot to {save_name}")

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "fidelity_encoder_ep9.pth"  # Note: encoder saves as .pth, not .pt
    DATA_PATH = "data/train.pt" # Or your specific dataset path
    
    # 1. Load Dataset
    print(f"Loading Dataset from {DATA_PATH}...")
    # Generate a small set if file doesn't exist just for testing logic
    dataset = EncoderDataset(file_path=DATA_PATH, size=50, verbose=True)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_quantum_graphs, shuffle=True)
    
    # 2. Initialize Model
    # Ensure these params match your training config!
    print("Initializing Model...")
    num_gate_types = len(get_universal_gate_set()['all'])
    model = FidelityEncoder(num_gate_types=num_gate_types, max_qubits=20).to(DEVICE)
    
    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found! Initializing random weights.")

    # 4. Run Evaluation
    verify_model(model, loader, DEVICE)
    visualize_latent_space(model, loader, DEVICE)