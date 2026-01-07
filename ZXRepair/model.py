import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv, global_mean_pool
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = DenseGCNConv(num_node_features, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNConv(hidden_dim, latent_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, adj, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        B, N, _ = x.shape
        
        h = self.conv1(x, adj, mask)
        h = h.view(B * N, -1)
        h = self.bn1(h)
        h = h.view(B, N, -1)
        h = F.relu(h)
        
        h = self.conv2(h, adj, mask)
        h = h.view(B * N, -1)
        h = self.bn2(h)
        h = h.view(B, N, -1)
        h = F.relu(h)
        
        h = self.conv3(h, adj, mask)
        h = F.relu(h)
        
        return h


class AdjacencyDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_embeddings):
        B, N, D = node_embeddings.shape
        
        h_i = node_embeddings.unsqueeze(2).expand(B, N, N, D)
        h_j = node_embeddings.unsqueeze(1).expand(B, N, N, D)
        
        pairs = torch.cat([h_i, h_j], dim=-1)
        
        edge_logits = self.edge_mlp(pairs).squeeze(-1)
        
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        adj_pred = torch.sigmoid(edge_logits)
        
        return adj_pred


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_node_features):
        super().__init__()
        self.feature_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features)
        )
        
    def forward(self, node_embeddings):
        return self.feature_mlp(node_embeddings)


class ZXRepairNet(nn.Module):
    def __init__(self, num_node_features=6, hidden_dim=128, latent_dim=256, 
                 max_nodes=256, lambda_adj=1.0, lambda_feat=1.0, lambda_gflow=0.5):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        
        self.lambda_adj = lambda_adj
        self.lambda_feat = lambda_feat
        self.lambda_gflow = lambda_gflow
        
        self.encoder = GraphEncoder(num_node_features, hidden_dim, latent_dim)
        
        self.adj_decoder = AdjacencyDecoder(latent_dim, hidden_dim)
        self.feat_decoder = FeatureDecoder(latent_dim, hidden_dim, num_node_features)
        
        self.gflow_encoder = None
        self.gflow_centroid = None
        self._load_gflow_encoder()
        
    def _load_gflow_encoder(self):
        try:
            from GflowEncoder.embedding_scorer import GraphEncoder as GflowGraphEncoder
            
            encoder_path = 'GflowEncoder/encoder.pth'
            centroid_path = 'GflowEncoder/valid_centroid.pt'
            
            if os.path.exists(encoder_path) and os.path.exists(centroid_path):
                self.gflow_encoder = GflowGraphEncoder(
                    num_node_features=6, hidden_dim=128, embedding_dim=64
                )
                self.gflow_encoder.load_state_dict(
                    torch.load(encoder_path, map_location='cpu', weights_only=True)
                )
                self.gflow_centroid = torch.load(
                    centroid_path, map_location='cpu', weights_only=True
                )
                
                for param in self.gflow_encoder.parameters():
                    param.requires_grad = False
                self.gflow_encoder.eval()
                
                print("Loaded GflowEncoder for validity loss")
        except Exception as e:
            print(f"Warning: Could not load GflowEncoder: {e}")
    
    def forward(self, x, adj, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
        
        node_embeddings = self.encoder(x, adj, mask)
        
        adj_pred = self.adj_decoder(node_embeddings)
        feat_pred = self.feat_decoder(node_embeddings)
        
        return adj_pred, feat_pred
    
    def compute_loss(self, x_corrupt, adj_corrupt, x_target, adj_target, mask=None):
        device = x_corrupt.device
        
        adj_pred, feat_pred = self.forward(x_corrupt, adj_corrupt, mask)
        
        loss_adj = F.binary_cross_entropy(adj_pred, adj_target, reduction='mean')
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            feat_diff = ((feat_pred - x_target) ** 2) * mask_expanded
            loss_feat = feat_diff.sum() / mask_expanded.sum().clamp(min=1)
        else:
            loss_feat = F.mse_loss(feat_pred, x_target)
        
        loss_gflow = torch.tensor(0.0, device=device)
        if self.gflow_encoder is not None and self.gflow_centroid is not None:
            self.gflow_encoder.eval()
            centroid = self.gflow_centroid.to(device)
            
            with torch.no_grad():
                embedding = self.gflow_encoder.forward_dense(x_corrupt, adj_pred)
            
            similarity = (embedding @ centroid.unsqueeze(1)).squeeze()
            score = (similarity + 1) / 2
            loss_gflow = (1 - score).mean()
        
        total_loss = (
            self.lambda_adj * loss_adj +
            self.lambda_feat * loss_feat +
            self.lambda_gflow * loss_gflow
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'adj': loss_adj.item(),
            'feat': loss_feat.item(),
            'gflow': loss_gflow.item() if isinstance(loss_gflow, torch.Tensor) else loss_gflow
        }
        
        return total_loss, loss_dict
    
    def repair(self, x, adj, threshold=0.5):
        self.eval()
        with torch.no_grad():
            adj_pred, feat_pred = self.forward(x, adj)
            
            adj_repaired = (adj_pred > threshold).float()
            
            adj_repaired = torch.max(adj_repaired, adj_repaired.transpose(-1, -2))
            
        return adj_repaired, feat_pred


class ZXNet(ZXRepairNet):
    pass