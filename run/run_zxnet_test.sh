#!/bin/bash
#SBATCH --job-name=zxnet_test
#SBATCH --output=logs/zxnet_test_%j.out
#SBATCH --error=logs/zxnet_test_%j.err
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

cd /home/wo057552/ALT-ZX

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

echo "=== ZXNet Test Training (2 epochs) ==="
# Quick test with 2 epochs
PYTHONPATH=. python -c "
import torch
from torch.utils.data import DataLoader, random_split
from ZXNet.model import ZXNet
from torch_geometric.data import Batch
import os

# Config
CHECKPOINT_PATH = 'ZXNet/model.pth'
DATASET_PATH = 'combined/dataset.pt'
FIDELITY_THRESHOLD = 0.7
BATCH_SIZE = 512
LR = 0.001
TEST_EPOCHS = 2

def collate_wrapper(batch):
    g1_list, g2_list, fidelities = [], [], []
    for sample in batch:
        g1, g2 = sample[0], sample[1]
        fid = sample[2] if len(sample) > 2 else 1.0
        g1_list.append(g1)
        g2_list.append(g2)
        fidelities.append(fid)
    batch_g1 = Batch.from_data_list(g1_list)
    batch_g2 = Batch.from_data_list(g2_list)
    return batch_g1, batch_g2, torch.tensor(fidelities)

print(f'Loading dataset from {DATASET_PATH}...')
dataset = torch.load(DATASET_PATH, weights_only=False)
print(f'Dataset size: {len(dataset)} samples')

# Split
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_wrapper, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

num_features = dataset[0][0].x.shape[1]
print(f'Node features: {num_features}')

model = ZXNet(num_node_features=num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'Starting test training for {TEST_EPOCHS} epochs...')
for epoch in range(TEST_EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_g1, batch_g2, labels_raw in train_loader:
        batch_g1 = batch_g1.to(device)
        batch_g2 = batch_g2.to(device)
        fidelity = labels_raw.to(device).float()
        
        optimizer.zero_grad()
        predictions = model((batch_g1, batch_g2))
        loss = model.zxnet_loss(predictions, fidelity, (batch_g1, batch_g2))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Quick val
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_g1, batch_g2, labels_raw in val_loader:
            batch_g1 = batch_g1.to(device)
            batch_g2 = batch_g2.to(device)
            fidelity = labels_raw.to(device).float()
            predictions = model((batch_g1, batch_g2))
            preds_binary = (predictions >= FIDELITY_THRESHOLD).long()
            labels_binary = (fidelity >= FIDELITY_THRESHOLD).long()
            val_correct += (preds_binary == labels_binary).sum().item()
            val_total += fidelity.size(0)
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    print(f'Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}')

print('Test training completed successfully!')
"
