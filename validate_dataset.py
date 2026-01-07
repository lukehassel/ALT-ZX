import torch
import numpy as np
from torch_geometric.data import Data
import sys


def validate_dataset(path='GflowNet/gflow_regression_dataset.pt'):
    print(f"=== Validating Dataset: {path} ===\n")
    
    try:
        dataset = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {path}")
        return False
    
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False
    
    sample = dataset[0]
    print(f"\n--- Sample Format ---")
    print(f"Keys: {list(sample.keys()) if isinstance(sample, dict) else 'PyG Data'}")
    
    if isinstance(sample, Data):
        print(f"  x shape: {sample.x.shape}")
        print(f"  edge_index shape: {sample.edge_index.shape}")
        print(f"  y: {sample.y}")
    elif isinstance(sample, dict):
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    
    print(f"\n--- Feature Check ---")
    if isinstance(sample, dict):
        x = sample['x']
        if x.shape[1] == 6:
            print(f"✓ 6 features per node (expected)")
            print(f"  Features: [VertexID, NodeType, Row, Degree, Phase, Qubit]")
        else:
            print(f"✗ Expected 6 features, got {x.shape[1]}")
    
    print(f"\n--- Label Distribution ---")
    labels = []
    for s in dataset:
        if isinstance(s, dict):
            labels.append(s['y'])
        else:
            labels.append(s.y.item())
    
    labels = np.array(labels)
    print(f"Min label: {labels.min():.1f}")
    print(f"Max label: {labels.max():.1f}")
    print(f"Mean label: {labels.mean():.2f}")
    print(f"Unique labels: {sorted(set(labels.astype(int)))}")
    
    for lb in sorted(set(labels.astype(int))):
        count = np.sum(labels == lb)
        pct = 100 * count / len(labels)
        print(f"  Label {int(lb):2d}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n--- Rule Sequence Check ---")
    if isinstance(sample, dict) and 'rule_sequence' in sample:
        print(f"✓ rule_sequence field present")
        has_rules = sum(1 for s in dataset if len(s.get('rule_sequence', [])) > 0)
        print(f"  Samples with rule sequences: {has_rules}/{len(dataset)}")
    else:
        print(f"✗ rule_sequence field not found")
    
    print(f"\n=== Validation Complete ===")
    
    if labels.max() > 0 and labels.min() == 0:
        print(f"✓ Dataset has varied labels (0 to {int(labels.max())})")
        return True
    elif labels.max() == labels.min():
        print(f"✗ WARNING: All labels are {int(labels.max())} (no variation)")
        return False
    else:
        return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'GflowNet/gflow_regression_dataset.pt'
    validate_dataset(path)
