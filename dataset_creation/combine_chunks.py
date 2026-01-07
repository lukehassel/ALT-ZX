#!/usr/bin/env python3

import os
import sys
import glob
import torch
import random
import hashlib
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from zx_loader import pyg_to_circuit
from qiskit import qasm2


CONFIG = {
    "chunk_dir": "/home/wo057552/ALT-ZX/optimization_data",
    "output_dir": "/home/wo057552/ALT-ZX/combined",
    "test_ratio": 0.2,
    "seed": 42,
    "no_shuffle": False,
}


def find_chunk_files(chunk_dir):
    pattern = os.path.join(chunk_dir, "*.pt")
    return sorted(glob.glob(pattern))


def compute_sample_hash(sample):
    try:
        data1, data2, _ = sample
        circuit1 = pyg_to_circuit(data1)
        circuit2 = pyg_to_circuit(data2)
        qasm1 = qasm2.dumps(circuit1)
        qasm2_str = qasm2.dumps(circuit2)
        combined = qasm1 + "|||" + qasm2_str
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception:
        return None


def load_all_chunks_simple(chunk_files, verbose=True):
    all_samples = []
    
    if verbose:
        print(f"Loading {len(chunk_files)} chunk files...")
    
    for chunk_file in tqdm(chunk_files, disable=not verbose):
        if os.path.getsize(chunk_file) == 0:
            print(f"Skipping empty file: {chunk_file}")
            continue
            
        try:
            chunk_data = torch.load(chunk_file, weights_only=False)
            if isinstance(chunk_data, list):
                all_samples.extend(chunk_data)
            else:
                all_samples.append(chunk_data)
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
            continue
    
    if verbose:
        print(f"Loaded {len(all_samples)} total samples.")
    
    return all_samples


def load_all_chunks_deduplicated(chunk_files, verbose=True):
    all_samples = []
    seen_hashes = set()
    duplicates_skipped = 0
    conversion_errors = 0
    
    if verbose:
        print(f"Loading {len(chunk_files)} chunk files with deduplication...")
    
    for chunk_file in tqdm(chunk_files, disable=not verbose):
        if os.path.getsize(chunk_file) == 0:
            print(f"Skipping empty file: {chunk_file}")
            continue
            
        try:
            chunk_data = torch.load(chunk_file, weights_only=False)
            samples = chunk_data if isinstance(chunk_data, list) else [chunk_data]
            
            for sample in samples:
                sample_hash = compute_sample_hash(sample)
                
                if sample_hash is None:
                    conversion_errors += 1
                    continue
                    
                if sample_hash in seen_hashes:
                    duplicates_skipped += 1
                    continue
                
                seen_hashes.add(sample_hash)
                all_samples.append(sample)
                
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
            continue
    
    if verbose:
        print(f"Loaded {len(all_samples)} unique samples.")
        print(f"Skipped {duplicates_skipped} duplicates.")
        if conversion_errors > 0:
            print(f"Skipped {conversion_errors} samples due to conversion errors.")
    
    return all_samples


def main():
    print(f"Using configuration: {CONFIG}")
    
    chunk_dir = CONFIG["chunk_dir"]
    output_dir = CONFIG["output_dir"]
    seed = CONFIG["seed"]
    no_shuffle = CONFIG["no_shuffle"]
    batch_size = CONFIG.get("batch_size", 50)
    
    if not os.path.isdir(chunk_dir):
        raise ValueError(f"Chunk directory does not exist: {chunk_dir}")
    
    output_dir = output_dir if output_dir else chunk_dir
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_files = find_chunk_files(chunk_dir)
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_dir}")
    
    print(f"Found {len(chunk_files)} chunk files.")
    print(f"Processing in batches of {batch_size} chunk files...")
    
    total_samples = 0
    output_files = []
    
    for batch_idx in range(0, len(chunk_files), batch_size):
        batch_files = chunk_files[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size
        
        print(f"\n[Batch {batch_num}] Processing {len(batch_files)} chunk files...")
        samples = load_all_chunks_simple(batch_files, verbose=True)
        
        if samples:
            output_path = os.path.join(output_dir, f"batch_{batch_num:04d}.pt")
            print(f"Saving {len(samples)} samples to {output_path}...")
            torch.save(samples, output_path)
            output_files.append(output_path)
            total_samples += len(samples)
            
            del samples
            import gc
            gc.collect()
    
    print(f"\n" + "="*60)
    print(f"Done!")
    print(f"Total samples: {total_samples}")
    print(f"Output files: {len(output_files)}")
    for f in output_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
