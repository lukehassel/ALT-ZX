import torch
import os
from tqdm import tqdm
import numpy as np
from qiskit.converters import circuit_to_dag

from circuit_utils import (
    create_random_circuit_with_universal_gates,
    dag_to_gate_and_edges,
    get_universal_gate_set
)

DATASET_ARGS = {
    'samples': 5000,
    'min_qubits': 3,
    'max_qubits': 8,
    'min_depth': 5,
    'max_depth': 20,
    'output_dir': './data',
    'filename': 'circuit_dataset.pt'
}

all_gates = sorted(get_universal_gate_set()['all'])
gate_to_idx = {gate: idx for idx, gate in enumerate(all_gates)}

def generate_circuit_data(num_qubits, depth):
    """
    Generates a single data sample: a random circuit converted to graph tensors.
    """
    circuit = create_random_circuit_with_universal_gates(
        num_qubits=num_qubits,
        depth=depth,
        seed=None
    )

    dag = circuit_to_dag(circuit)
    _, adjacency, _, gate_info = dag_to_gate_and_edges(dag)

    node_features = []
    
    for gate_name, qubits, params in gate_info:
        gate_idx = gate_to_idx.get(gate_name, 0)
        q0 = qubits[0]
        q1 = qubits[1] if len(qubits) > 1 else 0
        node_features.append([gate_idx, q0, q1])

    x = torch.tensor(node_features, dtype=torch.float32)

    rows, cols = torch.where(adjacency > 0)
    edge_index = torch.stack([rows, cols], dim=0).long()

    return {
        'x': x,
        'edge_index': edge_index,
        'num_qubits': num_qubits,
        'num_nodes': x.shape[0]
    }

def main():
    print(f"Global Gate Set ({len(all_gates)}): {all_gates}")
    
    samples = DATASET_ARGS['samples']
    min_qubits = DATASET_ARGS['min_qubits']
    max_qubits = DATASET_ARGS['max_qubits']
    min_depth = DATASET_ARGS['min_depth']
    max_depth = DATASET_ARGS['max_depth']
    output_dir = DATASET_ARGS['output_dir']
    filename = DATASET_ARGS['filename']
    
    args_dict = {
        **DATASET_ARGS,
        'num_gates': len(all_gates),
        'gate_set': all_gates
    }
    
    dataset = []
    
    print(f"Generating {samples} circuits...")
    
    for i in tqdm(range(samples)):
        n_qubits = np.random.randint(min_qubits, max_qubits + 1)
        depth = np.random.randint(min_depth, max_depth + 1)
        
        try:
            data = generate_circuit_data(n_qubits, depth)
            dataset.append(data)
        except Exception as e:
            print(f"Skipping circuit {i} due to error: {e}")
            continue

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    
    torch.save(dataset, save_path)
    
    map_path = os.path.join(output_dir, 'gate_map.pt')
    torch.save(gate_to_idx, map_path)
    
    args_path = os.path.join(output_dir, 'dataset_args.pt')
    torch.save(args_dict, args_path)
    
    print(f"\nSaved {len(dataset)} samples to {save_path}")
    print(f"Saved global gate map to {map_path}")
    print(f"Saved dataset arguments to {args_path}")

if __name__ == "__main__":
    main()