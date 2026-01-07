#!/usr/bin/env python3

import os
import sys
import hashlib
from tqdm import tqdm

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


CONFIG = {
    "input_file": "/home/wo057552/ALT-ZX/combined/dataset_mpo.pt",
    "output_file": "/home/wo057552/ALT-ZX/combined/dataset_mpo_dedup.pt",
}


def compute_sample_hash(sample):
    try:
        data1, data2, fidelity = sample
        parts = []
        
        if hasattr(data1, 'x') and data1.x is not None:
            parts.append(data1.x.numpy().tobytes())
        if hasattr(data1, 'edge_index') and data1.edge_index is not None:
            parts.append(data1.edge_index.numpy().tobytes())
        if hasattr(data2, 'x') and data2.x is not None:
            parts.append(data2.x.numpy().tobytes())
        if hasattr(data2, 'edge_index') and data2.edge_index is not None:
            parts.append(data2.edge_index.numpy().tobytes())
        
        combined = b'|||'.join(parts)
        return hashlib.md5(combined).hexdigest()
    except:
        return None


def main():
    print(f"Configuration: {CONFIG}")
    
    input_file = CONFIG["input_file"]
    output_file = CONFIG["output_file"]
    
    print(f"Loading {input_file}...")
    samples = torch.load(input_file)
    print(f"Loaded {len(samples)} samples.")
    
    seen_hashes = set()
    unique_samples = []
    duplicates = 0
    errors = 0
    
    for sample in tqdm(samples, desc="Deduplicating"):
        h = compute_sample_hash(sample)
        if h is None:
            errors += 1
            continue
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)
        unique_samples.append(sample)
    
    print(f"\nResults:")
    print(f"  Original: {len(samples)}")
    print(f"  Unique: {len(unique_samples)}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Errors: {errors}")
    
    print(f"\nSaving to {output_file}...")
    torch.save(unique_samples, output_file)
    print("Done!")


if __name__ == "__main__":
    main()
