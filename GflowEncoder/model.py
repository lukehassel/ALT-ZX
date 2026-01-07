import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features=6, hidden_dim=64, embedding_dim=64):
        super(GraphEncoder, self).__init__()
        
        self.input_dim = num_node_features - 1  # Skip node ID (feature 0)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.conv1 = GCNConv(self.input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.embed_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = x[:, 1:]  # Skip node ID
        
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = global_mean_pool(x, batch)
        embedding = self.embed_fc(x)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize for cosine distance
        
        return embedding
    
    def forward_dense(self, node_features, adj_matrix):
        # Dense adjacency forward pass for generator integration
        batch_size, num_nodes, _ = node_features.shape
        device = node_features.device
        
        embeddings = []
        for i in range(batch_size):
            x = node_features[i, :, 1:]  # Skip node ID
            A = adj_matrix[i]
            
            # Add self-loops (matches PyG's GCNConv default)
            A = A + torch.eye(num_nodes, device=device)
            
            # Symmetric normalization: D^(-1/2) @ A @ D^(-1/2) (matches PyG's GCNConv)
            deg = A.sum(dim=1).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            A_norm = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
            
            # GCN layers: X' = ReLU(A_norm @ X @ W + b)
            x = F.relu(A_norm @ x @ self.conv1.lin.weight.T + self.conv1.bias)
            x = F.relu(A_norm @ x @ self.conv2.lin.weight.T + self.conv2.bias)
            
            # Global pooling over active nodes only
            node_mask = node_features[i, :, 1:].abs().sum(dim=1) > 0.01
            if node_mask.sum() > 0:
                x_active = x[node_mask]
                x = x_active.mean(dim=0, keepdim=True)
            else:
                x = x.mean(dim=0, keepdim=True)
            
            emb = self.embed_fc(x)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
        
        return torch.cat(embeddings, dim=0)


def compute_valid_centroid(encoder, valid_graphs, device='cpu'):
    encoder.eval()
    encoder.to(device)
    
    loader = DataLoader(valid_graphs, batch_size=256, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_weight = batch.edge_attr.squeeze() if batch.edge_attr is not None else None
            emb = encoder(batch.x, batch.edge_index, batch.batch, edge_weight=edge_weight)
            embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    centroid = embeddings.mean(dim=0)
    centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    return centroid


def embedding_distance_score(encoder, graph_data, centroid, device='cpu'):
    encoder.eval()
    encoder.to(device)
    centroid = centroid.to(device)
    
    with torch.no_grad():
        graph_data = graph_data.to(device)
        edge_weight = graph_data.edge_attr.squeeze() if graph_data.edge_attr is not None else None
        emb = encoder(graph_data.x, graph_data.edge_index, edge_weight=edge_weight)
        
        # cosine_similarity returns [-1, 1], convert to [0, 1]
        similarity = F.cosine_similarity(emb, centroid.unsqueeze(0))
        score = (similarity + 1) / 2
    
    return score.item()


if __name__ == "__main__":
    print("Testing GraphEncoder...")
    
    encoder = GraphEncoder(num_node_features=6, hidden_dim=64, embedding_dim=32)
    
    x = torch.randn(10, 6)
    edge_index = torch.randint(0, 10, (2, 20))
    
    emb = encoder(x, edge_index)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding norm: {emb.norm().item():.4f}")
    
    print("✓ GraphEncoder test passed")
