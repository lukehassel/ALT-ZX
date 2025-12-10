import argparse
import os
import glob
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from tqdm import tqdm
import pyzx as zx

from model import ZXVGAE
from zx_loader import qasm_to_pyg, tensor_to_pyzx

class ZXQASMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = glob.glob(os.path.join(root_dir, '*.qasm'))
        self.valid_files = [f for f in self.files if os.path.getsize(f) > 0]
        super().__init__(root_dir, transform)

    def len(self):
        return len(self.valid_files)

    def get(self, idx):
        file_path = self.valid_files[idx]
        try:
            data = qasm_to_pyg(file_path)
            return data
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return qasm_to_pyg(self.valid_files[0])

def validate_gflow(model, loader, device, num_samples=5):
    model.eval()
    valid_circuits = 0
    total_checked = 0
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_samples: break
            
            data = data.to(device)
            z, edge_recon, node_logits, phase_pred = model(data.x, data.edge_index)
            
            adj_recon = (edge_recon > 0.5).float()
            node_types = node_logits.argmax(dim=1)
            node_types_onehot = torch.nn.functional.one_hot(node_types, num_classes=5)
            
            try:
                candidate_graph = tensor_to_pyzx(adj_recon, node_types_onehot, phase_pred)
                
                if zx.gflow.gflow(candidate_graph):
                    valid_circuits += 1
            except Exception as e:
                pass
            
            total_checked += 1
            
    return valid_circuits / max(total_checked, 1)

def train():
    parser = argparse.ArgumentParser(description='Train ZX-VGAE for Quantum Circuit Optimization')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing .qasm files')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='zx_vgae_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=True),
    ])

    print(f"Loading QASM files from {args.data_dir}...")
    dataset = ZXQASMDataset(args.data_dir, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ZXVGAE(in_channels=6, 
                   hidden_channels=args.hidden_dim, 
                   out_channels=args.latent_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            
            loss = model.loss(data.x, data.edge_index, 
                              data.pos_edge_label_index, 
                              data.neg_edge_label_index)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            gflow_score = validate_gflow(model, val_loader, device)
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Valid Gflow Rate: {gflow_score*100:.1f}% (Valid Quantum Circuits Generated)")
            
            torch.save(model.state_dict(), args.save_path)

    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    train()