import torch
from torch.utils.data import DataLoader
from ZXNet.model import ZXNet, PairData
import os
import glob
import pickle
import random
from zxnet_features import extract_zxnet_features

LR = 0.001
EPOCHS = 100
UNCERTAINTY_THRESH = -5.0
JACCARD_THRESH = 0.8
BATCH_SIZE = 2048
FIDELITY_THRESHOLD = 0.7
CHECKPOINT_PATH = 'ZXNet/model.pth'


def load_pickle_or_torch(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return torch.load(filepath, weights_only=False)


def evaluate_model(model, val_loader, device, print_confusion_matrix=False):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_batches = 0
    val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_g1, batch_g2, labels_raw in val_loader:
            batch_g1 = batch_g1.to(device, non_blocking=True)
            batch_g2 = batch_g2.to(device, non_blocking=True)
            fidelity = labels_raw.to(device, non_blocking=True).float()
            
            predictions = model((batch_g1, batch_g2))
            
            preds_binary = (predictions >= FIDELITY_THRESHOLD).long()
            labels_binary = (fidelity >= FIDELITY_THRESHOLD).long()
            
            loss_fn = model.module.zxnet_loss if hasattr(model, "module") else model.zxnet_loss
            loss = loss_fn(predictions, fidelity, (batch_g1, batch_g2))
            val_loss += loss.item()
            val_batches += 1
            
            val_correct += (preds_binary == labels_binary).sum().item()
            val_total += fidelity.size(0)
            
            val_tp += ((preds_binary == 1) & (labels_binary == 1)).sum().item()
            val_tn += ((preds_binary == 0) & (labels_binary == 0)).sum().item()
            val_fp += ((preds_binary == 1) & (labels_binary == 0)).sum().item()
            val_fn += ((preds_binary == 0) & (labels_binary == 1)).sum().item()
    
    avg_loss = val_loss / val_batches if val_batches > 0 else 0
    accuracy = val_correct / val_total if val_total > 0 else 0
    precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if print_confusion_matrix:
        print(f"Confusion Matrix (Threshold {FIDELITY_THRESHOLD}):")
        print(f"TP: {val_tp} | FN: {val_fn}")
        print(f"FP: {val_fp} | TN: {val_tn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': val_tp, 'tn': val_tn, 'fp': val_fp, 'fn': val_fn
    }


def balance_dataset(dataset, fidelity_threshold=0.7, optim_data_dir="optimization_data"):
    def normalize_sample(sample):
        for data in [sample[0], sample[1]]:
            if hasattr(data, 'num_qubits'):
                del data.num_qubits
        return sample
    
    dataset = [normalize_sample(s) for s in dataset]
    
    low_fidelity = [s for s in dataset if float(s[2]) < fidelity_threshold]
    high_fidelity_main = [s for s in dataset if float(s[2]) >= fidelity_threshold]
    
    print(f"Main dataset: {len(dataset)} total")
    print(f"  Low fidelity (<{fidelity_threshold}): {len(low_fidelity)}")
    print(f"  High fidelity (>={fidelity_threshold}): {len(high_fidelity_main)}")
    
    target_high = len(low_fidelity)
    needed_high = target_high - len(high_fidelity_main)
    
    if needed_high > 0:
        print(f"Need {needed_high} more high-fidelity samples to balance.")
        
        optim_files = sorted(glob.glob(os.path.join(optim_data_dir, "optim_chunk_*.pt")))
        print(f"Found {len(optim_files)} optimization chunk files.")
        
        high_fidelity_optim = []
        for f in optim_files:
            if len(high_fidelity_optim) >= needed_high:
                break
            try:
                chunk = load_pickle_or_torch(f)
                high_samples = [s for s in chunk if float(s[2]) >= fidelity_threshold]
                
                for s in high_samples:
                    if hasattr(s[0], 'num_qubits'):
                        del s[0].num_qubits
                    if hasattr(s[1], 'num_qubits'):
                        del s[1].num_qubits
                
                high_fidelity_optim.extend(high_samples)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                continue
        
        print(f"Loaded {len(high_fidelity_optim)} high-fidelity samples from optimization_data.")
        
        if len(high_fidelity_optim) > needed_high:
            random.shuffle(high_fidelity_optim)
            high_fidelity_optim = high_fidelity_optim[:needed_high]
        
        all_high_fidelity = high_fidelity_main + high_fidelity_optim
    else:
        print("Main dataset already has enough high-fidelity samples.")
        random.shuffle(high_fidelity_main)
        all_high_fidelity = high_fidelity_main[:target_high]
    
    balanced_dataset = low_fidelity + all_high_fidelity
    random.shuffle(balanced_dataset)
    
    expected_features = balanced_dataset[0][0].x.shape[1] if len(balanced_dataset) > 0 else 6
    filtered_dataset = []
    removed_count = 0
    for s in balanced_dataset:
        if s[0].x.shape[1] == expected_features and s[1].x.shape[1] == expected_features:
            filtered_dataset.append(s)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"Removed {removed_count} samples with mismatched feature dimensions")
    
    balanced_dataset = filtered_dataset
    
    print(f"\nBalanced dataset: {len(balanced_dataset)} samples")
    low_count = sum(1 for s in balanced_dataset if float(s[2]) < fidelity_threshold)
    high_count = len(balanced_dataset) - low_count
    print(f"  Low fidelity: {low_count} ({100*low_count/len(balanced_dataset):.1f}%)")
    print(f"  High fidelity: {high_count} ({100*high_count/len(balanced_dataset):.1f}%)")
    
    return balanced_dataset


if __name__ == "__main__":
    dataset_path = 'combined/dataset.pt'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        exit(1)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_pickle_or_torch(dataset_path)

    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        exit(1)
    
    dataset = balance_dataset(dataset, FIDELITY_THRESHOLD)
    
    sample = dataset[0]
    num_features = sample[0].x.shape[1]
    print(f"Dataset features: {num_features} (Expected: 6)")
    
    if num_features != 6:
        print(f"Warning: Expected 6 features, got {num_features}. Run migrate_dataset.py if needed.")
    
    from torch_geometric.data import Batch, Data

    def collate_wrapper(batch):
        g1_list = []
        g2_list = []
        labels_list = []
        
        for sample in batch:
            g1, g2, label = sample[0], sample[1], sample[2]
            g1_list.append(g1)
            g2_list.append(g2)
            labels_list.append(label)
        
        batch_g1 = Batch.from_data_list(g1_list)
        batch_g2 = Batch.from_data_list(g2_list)
        labels = torch.stack(labels_list)
        
        return batch_g1, batch_g2, labels

    from torch.utils.data import random_split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_wrapper, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_wrapper, num_workers=0, pin_memory=True)

    from torch_geometric.nn import DataParallel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ZXNet(num_node_features=num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded legacy checkpoint (no epoch info)")

    print("Starting training...")
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_g1, batch_g2, labels_raw in train_loader:
            batch_g1 = batch_g1.to(device, non_blocking=True)
            batch_g2 = batch_g2.to(device, non_blocking=True)
            labels_raw = labels_raw.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            predictions = model((batch_g1, batch_g2))
            fidelity = labels_raw.float()
            
            loss_fn = model.module.zxnet_loss if hasattr(model, "module") else model.zxnet_loss
            loss = loss_fn(predictions, fidelity, (batch_g1, batch_g2)) 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Loss {val_metrics['loss']:.4f} | Val Acc {val_metrics['accuracy']:.4f}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH} (epoch {epoch})")

    print("\n=== Final Validation Evaluation (Confusion Matrix) ===")
    evaluate_model(model, val_loader, device, print_confusion_matrix=True)