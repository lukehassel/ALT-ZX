import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ZXRepair.model import ZXRepairNet
from ZXRepair.dataset import load_dataset, generate_repair_dataset, save_dataset


CONFIG = {
    'dataset_path': 'ZXRepair/repair_dataset.pt',
    'checkpoint_path': 'ZXRepair/checkpoints/model.pth',
    'max_nodes': 256,
    'hidden_dim': 128,
    'latent_dim': 256,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 100,
    'val_split': 0.2,
    'lambda_adj': 1.0,
    'lambda_feat': 0.5,
    'lambda_gflow': 0.5,
    'num_samples': 50000,
}


class RepairDataset(Dataset):
    def __init__(self, pairs, max_nodes=256, num_node_features=6):
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features
        
        print(f"Processing {len(pairs)} pairs...")
        
        self.corrupted_x = []
        self.corrupted_adj = []
        self.target_x = []
        self.target_adj = []
        self.masks = []
        
        for corrupted, original in pairs:
            cx, cadj, cmask = self._pad_graph(corrupted)
            self.corrupted_x.append(cx)
            self.corrupted_adj.append(cadj)
            
            tx, tadj, tmask = self._pad_graph(original)
            self.target_x.append(tx)
            self.target_adj.append(tadj)
            
            self.masks.append(tmask)
        
        self.corrupted_x = torch.stack(self.corrupted_x)
        self.corrupted_adj = torch.stack(self.corrupted_adj)
        self.target_x = torch.stack(self.target_x)
        self.target_adj = torch.stack(self.target_adj)
        self.masks = torch.stack(self.masks)
        
        self.on_gpu = False
        if torch.cuda.is_available():
            try:
                print("Moving dataset to GPU...")
                self.corrupted_x = self.corrupted_x.cuda()
                self.corrupted_adj = self.corrupted_adj.cuda()
                self.target_x = self.target_x.cuda()
                self.target_adj = self.target_adj.cuda()
                self.masks = self.masks.cuda()
                self.on_gpu = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU memory insufficient, keeping data on CPU")
                    torch.cuda.empty_cache()
                else:
                    raise
        
        print(f"Dataset ready: {self.corrupted_x.shape[0]} samples (GPU: {self.on_gpu})")
    
    def _pad_graph(self, data):
        n_nodes = min(data.x.shape[0], self.max_nodes)
        
        x = torch.zeros(self.max_nodes, self.num_node_features)
        x[:n_nodes] = data.x[:n_nodes, :self.num_node_features]
        
        adj = torch.zeros(self.max_nodes, self.max_nodes)
        if data.edge_index.numel() > 0:
            edge_index = data.edge_index
            mask = (edge_index[0] < self.max_nodes) & (edge_index[1] < self.max_nodes)
            valid_edges = edge_index[:, mask]
            if valid_edges.numel() > 0:
                adj[valid_edges[0], valid_edges[1]] = 1.0
        
        adj.fill_diagonal_(1.0)
        
        node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        node_mask[:n_nodes] = True
        
        return x, adj, node_mask
    
    def __len__(self):
        return self.corrupted_x.shape[0]
    
    def __getitem__(self, idx):
        return {
            'corrupted_x': self.corrupted_x[idx],
            'corrupted_adj': self.corrupted_adj[idx],
            'target_x': self.target_x[idx],
            'target_adj': self.target_adj[idx],
            'mask': self.masks[idx],
        }


def train_epoch(model, dataloader, optimizer, device, data_on_gpu=False):
    model.train()
    total_loss = 0
    loss_components = {'adj': 0, 'feat': 0, 'gflow': 0}
    num_batches = 0
    
    for batch in dataloader:
        x_corrupt = batch['corrupted_x'] if data_on_gpu else batch['corrupted_x'].to(device)
        adj_corrupt = batch['corrupted_adj'] if data_on_gpu else batch['corrupted_adj'].to(device)
        x_target = batch['target_x'] if data_on_gpu else batch['target_x'].to(device)
        adj_target = batch['target_adj'] if data_on_gpu else batch['target_adj'].to(device)
        mask = batch['mask'] if data_on_gpu else batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(
            x_corrupt, adj_corrupt, x_target, adj_target, mask
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0)
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in loss_components:
        loss_components[k] /= num_batches if num_batches > 0 else 1
    
    return avg_loss, loss_components


def validate(model, dataloader, device, data_on_gpu=False):
    model.eval()
    total_loss = 0
    loss_components = {'adj': 0, 'feat': 0, 'gflow': 0}
    num_batches = 0
    
    edge_correct = 0
    edge_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x_corrupt = batch['corrupted_x'] if data_on_gpu else batch['corrupted_x'].to(device)
            adj_corrupt = batch['corrupted_adj'] if data_on_gpu else batch['corrupted_adj'].to(device)
            x_target = batch['target_x'] if data_on_gpu else batch['target_x'].to(device)
            adj_target = batch['target_adj'] if data_on_gpu else batch['target_adj'].to(device)
            mask = batch['mask'] if data_on_gpu else batch['mask'].to(device)
            
            loss, loss_dict = model.compute_loss(
                x_corrupt, adj_corrupt, x_target, adj_target, mask
            )
            
            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, 0)
            num_batches += 1
            
            adj_pred, _ = model(x_corrupt, adj_corrupt, mask)
            pred_binary = (adj_pred > 0.5).float()
            edge_correct += (pred_binary == adj_target).sum().item()
            edge_total += adj_target.numel()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in loss_components:
        loss_components[k] /= num_batches if num_batches > 0 else 1
    
    edge_acc = edge_correct / edge_total if edge_total > 0 else 0
    
    return avg_loss, loss_components, edge_acc


def main():
    parser = argparse.ArgumentParser(description='Train ZXRepairNet')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--num_samples', type=int, default=CONFIG['num_samples'],
                        help='Number of samples to generate if dataset not found')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if os.path.exists(CONFIG['dataset_path']):
        print(f"Loading dataset from {CONFIG['dataset_path']}...")
        pairs = load_dataset(CONFIG['dataset_path'])
    else:
        print(f"Dataset not found. Generating {args.num_samples} samples...")
        pairs = generate_repair_dataset(num_samples=args.num_samples, verbose=True)
        save_dataset(pairs, CONFIG['dataset_path'])
    
    print(f"Dataset size: {len(pairs)} pairs")
    
    dataset = RepairDataset(pairs, max_nodes=CONFIG['max_nodes'])
    
    val_size = int(len(dataset) * CONFIG['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
    )
    
    model = ZXRepairNet(
        num_node_features=6,
        hidden_dim=CONFIG['hidden_dim'],
        latent_dim=CONFIG['latent_dim'],
        max_nodes=CONFIG['max_nodes'],
        lambda_adj=CONFIG['lambda_adj'],
        lambda_feat=CONFIG['lambda_feat'],
        lambda_gflow=CONFIG['lambda_gflow'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    os.makedirs(os.path.dirname(CONFIG['checkpoint_path']), exist_ok=True)
    
    if args.resume and os.path.exists(CONFIG['checkpoint_path']):
        print(f"Loading checkpoint from {CONFIG['checkpoint_path']}...")
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    print("\nStarting training...")
    data_on_gpu = dataset.on_gpu
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device, data_on_gpu)
        
        val_loss, val_components, edge_acc = validate(model, val_loader, device, data_on_gpu)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} (adj={train_components['adj']:.4f}, "
              f"feat={train_components['feat']:.4f}, gflow={train_components['gflow']:.4f}) | "
              f"Val: {val_loss:.4f} | Edge Acc: {edge_acc:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, CONFIG['checkpoint_path'])
        
        if is_best:
            best_path = CONFIG['checkpoint_path'].replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  -> New best model saved!")
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()