import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GflowEncoder.model import GraphEncoder, compute_valid_centroid
from GflowEncoder.data_loader import create_data_loader
from zx_loader import pyzx_graph_to_pyg

# Triplet loss with cosine distance
triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
    margin=0.3
)


def train_encoder(
    encoder,
    data_loader,
    num_epochs=100,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    encoder = encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/100)
    
    print(f"Training GraphEncoder on {device}")
    print(f"Epochs: {num_epochs}, Batches per epoch: {len(data_loader)}")
    print(f"Initial LR: {lr}, Final LR: {lr/100:.2e}")
    
    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for anchor_batch, pos_batch, neg_batch in pbar:
            anchor_batch = anchor_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = encoder.forward(anchor_batch.x, anchor_batch.edge_index, anchor_batch.batch, edge_weight=anchor_batch.edge_attr.squeeze())
            pos_emb = encoder.forward(pos_batch.x, pos_batch.edge_index, pos_batch.batch, edge_weight=pos_batch.edge_attr.squeeze())
            neg_emb = encoder.forward(neg_batch.x, neg_batch.edge_index, neg_batch.batch, edge_weight=neg_batch.edge_attr.squeeze())
            
            loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
        
        scheduler.step()
    
    return encoder


def save_encoder(encoder, save_dir='GflowEncoder'):
    encoder_path = os.path.join(save_dir, 'encoder.pth')
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Saved encoder to {encoder_path}")


if __name__ == "__main__":
    data_loader = create_data_loader(
        dataset_path="GflowEncoder/dataset.pt",
        batch_size=64,
        shuffle=True
    )
    
    encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64)
    encoder = train_encoder(encoder, data_loader, num_epochs=100)
    save_encoder(encoder)
    
    print("\n✓ Training complete!")
    print("  Encoder: GflowEncoder/encoder.pth")
    print("  Run 'python -m GflowEncoder.compute_centroid' to compute the centroid.")
