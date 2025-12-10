#!/usr/bin/env python3
"""
Script to load and inspect a combined dataset file (train.pt or test.pt).
Prints the first N items in a readable format.
"""

import os
import torch
import json




def main():
    # Configuration dictionary
    config = {
        "dataset_file": "data/train.pt",  # Path to dataset file
        "num_samples": 10,  # Number of samples to print
        "start_idx": 0,    # Starting index
    }
    
    data = torch.load("data/train.pt")
    
    for item in data:
        print(item)


if __name__ == "__main__":
    main()

