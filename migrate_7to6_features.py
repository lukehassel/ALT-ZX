#!/usr/bin/env python
import pickle
import torch
from tqdm import tqdm


def convert_7feat_to_6feat(x):
    # 7-feat: [IsZ, IsX, IsBoundary, Phase, Row, Qubit, Degree]
    # 6-feat: [ID, Type, Row, Degree, Phase, Qubit]
    num_nodes = x.shape[0]
    new_x = torch.zeros(num_nodes, 6, dtype=x.dtype)
    
    for i in range(num_nodes):
        new_x[i, 0] = float(i)
        
        if x[i, 0] > 0.5:
            new_x[i, 1] = 1.0
        elif x[i, 1] > 0.5:
            new_x[i, 1] = 2.0
        else:
            new_x[i, 1] = 0.0
        
        new_x[i, 2] = x[i, 4]
        new_x[i, 3] = x[i, 6]
        new_x[i, 4] = x[i, 3]
        new_x[i, 5] = x[i, 5]
    
    return new_x


def migrate_graph(g):
    if hasattr(g, 'x') and g.x.shape[1] == 7:
        g.x = convert_7feat_to_6feat(g.x)
    return g


def main():
    input_path = 'combined/dataset.pt'
    output_path = 'combined/dataset_6feat.pt'
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    feat_6 = 0
    feat_7 = 0
    for sample in data:
        g1, g2 = sample[0], sample[1]
        if g1.x.shape[1] == 6:
            feat_6 += 1
        elif g1.x.shape[1] == 7:
            feat_7 += 1
    print(f"Before: {feat_6} 6-feat, {feat_7} 7-feat samples")
    
    print("Migrating 7-feat to 6-feat...")
    migrated = 0
    for sample in tqdm(data, desc="Migrating"):
        g1, g2 = sample[0], sample[1]
        
        if g1.x.shape[1] == 7:
            sample[0].x = convert_7feat_to_6feat(g1.x)
            migrated += 1
        if g2.x.shape[1] == 7:
            sample[1].x = convert_7feat_to_6feat(g2.x)
    
    feat_6_after = 0
    feat_7_after = 0
    for sample in data:
        g1, g2 = sample[0], sample[1]
        if g1.x.shape[1] == 6:
            feat_6_after += 1
        elif g1.x.shape[1] == 7:
            feat_7_after += 1
    print(f"After: {feat_6_after} 6-feat, {feat_7_after} 7-feat samples")
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Done! Migrated {migrated} samples.")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
