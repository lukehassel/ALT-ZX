import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

from GenZX.layers import MLP_VAE_plain, GNN_VAE

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ZXNet.model import ZXNet


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum',
                 zxnet_checkpoint='ZXNet/model.pth', lambda_zxnet=200.0, node_feature_dim=6):
        super(GraphVAE, self).__init__()
        
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        
        self.vae = GNN_VAE(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_num_nodes=max_num_nodes,
            output_dim=output_dim
        )
        self.node_feature_dim = node_feature_dim
        self.max_num_nodes = max_num_nodes
        self.pool = pool
        self.lambda_zxnet = lambda_zxnet

        self.zxnet = ZXNet(num_node_features=6)
        if os.path.exists(zxnet_checkpoint):
            ckpt = torch.load(zxnet_checkpoint, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.zxnet.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                self.zxnet.load_state_dict(ckpt, strict=False)
            print(f"Loaded ZXNet from {zxnet_checkpoint}")
        
        for param in self.zxnet.parameters():
            param.requires_grad = False
        self.zxnet.eval()
        
        from GflowEncoder.model import GraphEncoder
        self.gflow_encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64)
        encoder_path = 'GflowEncoder/encoder.pth'
        centroid_path = 'GflowEncoder/valid_centroid.pt'
        
        self.gflow_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
        self.gflow_centroid = torch.load(centroid_path, map_location='cpu', weights_only=True)
        print(f"Loaded GflowEncoder from {encoder_path}")
            
        for param in self.gflow_encoder.parameters():
            param.requires_grad = False
        self.gflow_encoder.eval()
        self.lambda_gflow = 5.0

    def recover_adj_lower(self, l):
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj
    
    def forward(self, input_features, adj, node_features=None):
        batch_size = adj.shape[0]
        device = adj.device
        
        if node_features is None:
            raise ValueError("node_features is required for GNN-based VAE encoder")
        
        h_decode, z_mu, z_lsgms = self.vae(node_features, adj)
        out = torch.sigmoid(h_decode)
        
        recon_adj = self.recover_adj_differentiable_batch(out)
        
        self.gflow_encoder.eval()
        
        if self.gflow_centroid is not None:
            centroid = self.gflow_centroid.to(device)
            
            # Use soft adjacency directly (GflowEncoder works with continuous values)
            embedding = self.gflow_encoder.forward_dense(node_features, recon_adj)
            
            similarity = (embedding @ centroid.unsqueeze(1)).squeeze()
            score = (similarity + 1) / 2
            
            loss_gflow = (1 - score).mean()
        else:
            loss_gflow = torch.tensor(0.0, device=device)
        
        loss_boundary = torch.tensor(0.0, device=device)
        if node_features is not None:
            loss_boundary = self.boundary_degree_loss(recon_adj, node_features)
        
        loss_recon = torch.nn.functional.binary_cross_entropy(
            recon_adj, 
            adj.float(),
            reduction='mean'
        )
        lambda_recon = 10.0
        
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= batch_size * self.max_num_nodes * self.max_num_nodes
        lambda_kl = 0.01
        
        p = recon_adj
        epsilon = 1e-6
        entropy = -p * torch.log(p + epsilon) - (1 - p) * torch.log(1 - p + epsilon)
        loss_entropy = entropy.mean()
        lambda_entropy = 0.1
        
        loss_zxnet = torch.tensor(0.0, device=device)
        if node_features is not None:
            loss_zxnet = self.zxnet_loss_batch(adj, recon_adj, node_features)

        lambda_boundary = 5.0
        
        loss = (lambda_recon * loss_recon +
                lambda_kl * loss_kl + 
                self.lambda_gflow * loss_gflow + 
                self.lambda_zxnet * loss_zxnet +
                lambda_boundary * loss_boundary + 
                lambda_entropy * loss_entropy)
        
        return loss
    
    def boundary_degree_loss(self, recon_adj, node_features):
        # Boundary nodes (NodeType=0) must have degree 1
        boundary_mask = torch.abs(node_features[:, :, 1] - 0.0) < 0.1
        
        if boundary_mask.sum() == 0:
            return torch.tensor(0.0, device=recon_adj.device)
            
        degrees = recon_adj.sum(dim=2)
        boundary_degrees = degrees[boundary_mask]
        
        loss = torch.nn.functional.mse_loss(
            boundary_degrees, 
            torch.ones_like(boundary_degrees),
            reduction='mean'
        )
        
        return loss
    
    def recover_adj_differentiable_batch(self, out):
        batch_size = out.shape[0]
        n = self.max_num_nodes
        device = out.device
        
        triu_indices = torch.triu_indices(n, n, device=device)
        
        adj = torch.zeros(batch_size, n, n, device=device, dtype=out.dtype)
        adj[:, triu_indices[0], triu_indices[1]] = out
        
        adj = adj + adj.transpose(1, 2) - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))
        
        return adj
    
    def zxnet_loss_batch(self, input_adj, recon_adj, input_features):
        with torch.no_grad():
            emb_input = self.zxnet.forward_one_graph_dense(
                input_features.detach(),
                input_adj.detach()
            )
        
        emb_recon = self.zxnet.forward_one_graph_dense(
            input_features.detach(),
            recon_adj
        )
        
        return F.mse_loss(emb_recon, emb_input)

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_truth, adj_pred)
