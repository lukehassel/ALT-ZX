#!/usr/bin/env python3

import os
import glob
import torch
import random
import argparse
from tqdm import tqdm


def find_chunk_files(chunk_dir):
    """Find all chunk files in the directory, sorted by index."""
    pattern = os.path.join(chunk_dir, "dataset_chunk_*.pt")
    chunk_files = sorted(glob.glob(pattern))
    return chunk_files


def load_all_chunks(chunk_files, verbose=True):
    """Load all chunk files and combine into a single list."""
    all_samples = []
    
    if verbose:
        print(f"Loading {len(chunk_files)} chunk files...")
    
    for chunk_file in tqdm(chunk_files, disable=not verbose):
        chunk_data = torch.load(chunk_file)
        if isinstance(chunk_data, list):
            all_samples.extend(chunk_data)
        else:
            all_samples.append(chunk_data)
    
    if verbose:
        print(f"Loaded {len(all_samples)} total samples.")
    
    return all_samples


def split_dataset(samples, test_ratio=0.2, shuffle=True, seed=42):
    """Split dataset into train and test sets."""
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    return train_samples, test_samples


def main():
    parser = argparse.ArgumentParser(
        description="Combine dataset chunks into train/test splits"
    )
    parser.add_argument(
        "--chunk_dir",
        type=str,
        required=True,
        help="Directory containing dataset chunk files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for train.pt and test.pt (default: same as chunk_dir)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle before splitting"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.chunk_dir):
        raise ValueError(f"Chunk directory does not exist: {args.chunk_dir}")
    
    if not (0 < args.test_ratio < 1):
        raise ValueError(f"test_ratio must be between 0 and 1, got {args.test_ratio}")
    
    output_dir = args.output_dir if args.output_dir else args.chunk_dir
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_files = find_chunk_files(args.chunk_dir)
    if not chunk_files:
        raise ValueError(f"No chunk files found in {args.chunk_dir}")
    
    print(f"Found {len(chunk_files)} chunk files.")
    all_samples = load_all_chunks(chunk_files, verbose=True)
    
    print(f"\nSplitting dataset (test_ratio={args.test_ratio}, shuffle={not args.no_shuffle})...")
    train_samples, test_samples = split_dataset(
        all_samples,
        test_ratio=args.test_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    train_path = os.path.join(output_dir, "train.pt")
    test_path = os.path.join(output_dir, "test.pt")
    
    print(f"\nSaving train set to {train_path}...")
    torch.save(train_samples, train_path)
    
    print(f"Saving test set to {test_path}...")
    torch.save(test_samples, test_path)
    
    print("\nDone!")
    print(f"Train set: {train_path} ({len(train_samples)} samples)")
    print(f"Test set: {test_path} ({len(test_samples)} samples)")


if __name__ == "__main__":
    main()

