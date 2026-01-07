#!/usr/bin/env python3

import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, random_split
from ZXNet.model import ZXNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

CHECKPOINT_PATH = 'ZXNet/model.pth'
DATASET_PATH = 'combined/dataset.pt'
FIDELITY_THRESHOLD = 0.7
BATCH_SIZE = 512


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


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Model not found at {CHECKPOINT_PATH}")
        return
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Dataset size: {len(dataset)} samples")
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_wrapper,
        num_workers=0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sample_g1, sample_g2, _ = dataset[0][0], dataset[0][1], dataset[0][2]
    num_features = sample_g1.x.shape[1]
    print(f"Node features: {num_features}")
    
    model = ZXNet(num_node_features=num_features).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint")
    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch_g1, batch_g2, labels_raw in val_loader:
            batch_g1 = batch_g1.to(device)
            batch_g2 = batch_g2.to(device)
            fidelity = labels_raw.to(device).float()
            
            predictions = model((batch_g1, batch_g2))
            
            preds_binary = (predictions >= FIDELITY_THRESHOLD).long()
            labels_binary = (fidelity >= FIDELITY_THRESHOLD).long()
            
            loss = model.zxnet_loss(predictions, fidelity, (batch_g1, batch_g2))
            val_loss += loss.item()
            
            val_correct += (preds_binary == labels_binary).sum().item()
            val_total += fidelity.size(0)
            
            val_tp += ((preds_binary == 1) & (labels_binary == 1)).sum().item()
            val_tn += ((preds_binary == 0) & (labels_binary == 0)).sum().item()
            val_fp += ((preds_binary == 1) & (labels_binary == 0)).sum().item()
            val_fn += ((preds_binary == 0) & (labels_binary == 1)).sum().item()
    
    accuracy = val_correct / val_total if val_total > 0 else 0
    precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("ZXNet Validation Results")
    print("="*50)
    print(f"Dataset:      {len(dataset)} total samples")
    print(f"Validation:   {val_total} samples")
    print(f"Threshold:    {FIDELITY_THRESHOLD}")
    print("-"*50)
    print(f"Accuracy:     {accuracy*100:.2f}%")
    print(f"Precision:    {precision*100:.2f}%")
    print(f"Recall:       {recall*100:.2f}%")
    print(f"F1 Score:     {f1*100:.2f}%")
    print("-"*50)
    print("Confusion Matrix:")
    print(f"  TP: {val_tp:6d} | FN: {val_fn:6d}")
    print(f"  FP: {val_fp:6d} | TN: {val_tn:6d}")
    print("="*50)


if __name__ == "__main__":
    main()
