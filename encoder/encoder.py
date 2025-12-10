import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv, GlobalAttentionPooling
from torch.utils.data import DataLoader, Dataset
from qiskit import QuantumCircuit
import os
import glob
import os
import glob
from mpo.circuit_utils import get_universal_gate_set
from encoder.utils import qasm_to_dgl

GATE_TO_IDX = {g: i for i, g in enumerate(get_universal_gate_set()['all'])}
IDX_TO_GATE = {i: g for g, i in GATE_TO_IDX.items()}
NUM_GATE_TYPES = len(get_universal_gate_set()['all'])


def graph_collate_fn(batch):
    """
    Takes a batch of dicts (QASM strings) and converts them to Batched DGL Graphs.
    """
    graphs_1, types_1, locs_1 = [], [], []
    graphs_2, types_2, locs_2 = [], [], []
    valid_indices = []
    
    for i, item in enumerate(batch):
        if item['fidelity'] < 0.8: 
            continue
            
        g1, t1, l1 = qasm_to_dgl(item['circuit_1'])
        g2, t2, l2 = qasm_to_dgl(item['circuit_2'])
        
        graphs_1.append(g1); types_1.append(t1); locs_1.append(l1)
        graphs_2.append(g2); types_2.append(t2); locs_2.append(l2)
    
    if not graphs_1:
        return None

    batched_g1 = dgl.batch(graphs_1)
    batched_t1 = torch.cat(types_1)
    batched_l1 = torch.cat(locs_1)
    
    batched_g2 = dgl.batch(graphs_2)
    batched_t2 = torch.cat(types_2)
    batched_l2 = torch.cat(locs_2)
    
    return (batched_g1, batched_t1, batched_l1), (batched_g2, batched_t2, batched_l2)


class FidelityEncoder(nn.Module):
    def __init__(self, num_gate_types, max_qubits=20, hidden_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        self.gate_embedding = nn.Embedding(num_gate_types, hidden_dim)
        self.qubit_embedding = nn.Embedding(max_qubits, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, num_heads=num_heads, 
                        feat_drop=0.1, attn_drop=0.1, residual=True, allow_zero_in_degree=True)
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.pooling = GlobalAttentionPooling(self.pool_gate)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64)
        )

    def forward(self, g, gate_types, qubit_indices=None, return_nodes=False):
        h = self.gate_embedding(gate_types)
        
        if qubit_indices is not None:
            h_loc = self.qubit_embedding(qubit_indices)
            h = h + h_loc 
        
        for layer in self.layers:
            h = layer(g, h).flatten(1) 
            h = F.relu(h)
            
        h = self.norm(h)
        
        if return_nodes:
            return h

        z_graph = self.pooling(g, h)
        z_proj = self.projection(z_graph)
        return z_proj

class CircuitInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        features = torch.cat([z_i, z_j], dim=0)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix), 1, 
            torch.arange(2 * batch_size).view(-1, 1).to(features.device), 0
        )
        
        labels = torch.cat([
            torch.arange(batch_size) + batch_size, 
            torch.arange(batch_size)
        ]).to(features.device)
        
        logits = similarity_matrix / self.temperature
        
        return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 10

    dataset = torch.load("data/train.pt")
    
    print(f"Loaded {len(dataset)} samples")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=graph_collate_fn, num_workers=0)

    model = FidelityEncoder(num_gate_types=NUM_GATE_TYPES, max_qubits=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = CircuitInfoNCELoss(temperature=0.07).to(DEVICE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        model.train()
        
        for batch_data in train_loader:
            if batch_data is None: print("Empty batch"); continue
            
            (g1, t1, l1), (g2, t2, l2) = batch_data
            
            g1, t1, l1 = g1.to(DEVICE), t1.to(DEVICE), l1.to(DEVICE)
            g2, t2, l2 = g2.to(DEVICE), t2.to(DEVICE), l2.to(DEVICE)
            
            optimizer.zero_grad()
            
            z1 = model(g1, t1, l1)
            z2 = model(g2, t2, l2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                print(f"Ep {epoch} Step {steps} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ===")
        
        torch.save(model.state_dict(), f"fidelity_encoder_ep{epoch}.pth")