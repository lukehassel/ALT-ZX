import torch
import pyzx as zx
import random
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GflowEncoder.model import GraphEncoder, compute_valid_centroid
from circuit_utils import create_random_circuit_with_universal_gates
from zx_loader import pyzx_graph_to_pyg
from qiskit import qasm2


def generate_valid_graphs(num_graphs=500):
    print(f"Generating {num_graphs} valid graphs...")
    valid_graphs = []
    errors = 0
    
    for i in tqdm(range(num_graphs * 2)):
        if len(valid_graphs) >= num_graphs:
            break
            
        seed = i * 7
        num_qubits = random.randint(3, 10)
        depth = random.randint(5, 10)
        
        try:
            qc = create_random_circuit_with_universal_gates(num_qubits, depth, seed=seed)
            circ = zx.Circuit.from_qasm(qasm2.dumps(qc))
            graph = circ.to_graph()
            
            data = pyzx_graph_to_pyg(graph)
            if data is not None:
                valid_graphs.append(data)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Exception {errors}: {type(e).__name__}: {e}")
    
    print(f"Generated {len(valid_graphs)} valid graphs ({errors} errors)")
    
    if len(valid_graphs) == 0:
        raise RuntimeError("Failed to generate any valid graphs. Check pyzx_graph_to_pyg.")
    
    return valid_graphs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    encoder_path = "GflowEncoder/encoder.pth"
    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found. Train the model first.")
        return
    
    encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    print(f"Loaded encoder from {encoder_path}")
    
    valid_graphs = generate_valid_graphs(500)
    
    print("Computing centroid...")
    centroid = compute_valid_centroid(encoder, valid_graphs, device=device)
    
    centroid_path = "GflowEncoder/valid_centroid.pt"
    torch.save(centroid, centroid_path)
    print(f"Saved centroid to {centroid_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
