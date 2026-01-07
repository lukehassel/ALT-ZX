import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from GenZX.model import GraphVAE
from GenZX.data_loader import load_dataset, GraphVAEDataset

CUDA = 2

LR_milestones = [25, 50, 100, 150, 200, 250, 300]

CONFIG = {
    'dataset': 'GenZX/genzx_dataset.pt',
    'lr': 0.001,
    'batch_size': 512,
    'num_workers': 0,
    'max_num_nodes': 64,
    'epochs': 1000,
}


def train(config, dataloader, model):
    os.makedirs('GenZX/checkpoints', exist_ok=True)
    
    optimizer = optim.Adam(list(model.parameters()), lr=config['lr'])
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=0.1)
    
    start_epoch = 0
    checkpoint_path = 'GenZX/checkpoints/model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch")

    model.train()
    for epoch in range(start_epoch, config['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            features = data['features'].float()
            adj_input = data['adj'].float()
            node_features = data['node_features'].float()
            
            loss = model(features, adj_input, node_features)
            
            if batch_idx == 0:
                with torch.no_grad():
                    h_decode, z_mu, z_lsgms = model.vae(node_features, adj_input)
                    out = torch.sigmoid(h_decode)
                    recon_adj = model.recover_adj_differentiable_batch(out)
                    embedding = model.gflow_encoder.forward_dense(node_features, recon_adj)
                    if model.gflow_centroid is not None:
                        centroid = model.gflow_centroid.to(recon_adj.device)
                        similarity = (embedding @ centroid.unsqueeze(1)).squeeze()
                        score = (similarity + 1) / 2
                        print(f"  [DEBUG] Epoch {epoch} Batch 0:")
                        print(f"    recon_adj mean: {recon_adj.mean():.4f}, range: [{recon_adj.min():.4f}, {recon_adj.max():.4f}]")
                        print(f"    embedding_score: {score.mean():.6f}")
                        print(f"    loss_gflow would be: {(1 - score).mean():.6f}")
                        
                        loss_boundary = model.boundary_degree_loss(recon_adj, node_features)
                        print(f"    loss_boundary: {loss_boundary.item():.6f}")
                        
                        loss_zxnet = model.zxnet_loss_batch(adj_input, recon_adj, node_features)
                        print(f"    loss_zxnet: {loss_zxnet.item():.6f}")

                        epsilon = 1e-6
                        entropy = -recon_adj * torch.log(recon_adj + epsilon) - (1 - recon_adj) * torch.log(1 - recon_adj + epsilon)
                        loss_entropy = entropy.mean()
                        print(f"    loss_entropy: {loss_entropy.item():.6f}")
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f'Epoch: {epoch}, Avg Loss: {avg_loss:.6f}')
    
        checkpoint_path = 'GenZX/checkpoints/model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)
    
    raw_data = load_dataset(CONFIG['dataset'])
    print(f'Loaded {len(raw_data)} samples')
    
    max_nodes = CONFIG['max_num_nodes']
    graphs = []
    for sample in raw_data:
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            g1, g2 = sample[0], sample[1]
            if hasattr(g1, 'x') and g1.x is not None and g1.x.shape[0] < max_nodes:
                graphs.append(g1)
            if hasattr(g2, 'x') and g2.x is not None and g2.x.shape[0] < max_nodes:
                graphs.append(g2)
        elif hasattr(sample, 'x') and sample.x is not None:
            if sample.x.shape[0] < max_nodes:
                graphs.append(sample)

    if max_nodes == -1:
        max_nodes = max([g.x.shape[0] for g in graphs])

    print(f'Total graphs: {len(graphs)}, max nodes: {max_nodes}')

    dataset = GraphVAEDataset(graphs, max_nodes)
    dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=CONFIG['batch_size'], 
            num_workers=CONFIG['num_workers'],
            shuffle=True)
    
    print(f"Initializing model with max_num_nodes={max_nodes}")

    model = GraphVAE(input_dim=max_nodes, 
                     hidden_dim=64, 
                     latent_dim=1024, 
                     max_num_nodes=max_nodes).cuda()
    
    train(CONFIG, dataset_loader, model)


if __name__ == '__main__':
    main()