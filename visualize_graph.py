import torch
import sys
import os
import matplotlib.pyplot as plt
import pyzx as zx

sys.path.insert(0, os.getcwd())
from zx_loader import reconstruct_pyzx_from_6feat


def main():
    dataset_path = 'combined/dataset.pt'
    print(f"Loading {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)
    
    data = dataset[0][0]
    
    print(f"Reconstructing graph with {data.x.shape[0]} nodes...")
    g = reconstruct_pyzx_from_6feat(data)
    
    print("Drawing graph...")
    fig = plt.figure(figsize=(12, 6))
    zx.draw_matplotlib(g, labels=True, h_edge_draw='blue')
    
    output_file = "zx_graph_155_nodes.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {output_file}")


if __name__ == "__main__":
    main()
