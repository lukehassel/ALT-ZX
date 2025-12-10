import torch
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeJakarta
import sys
import os

# Add the quest example directory to the path
quest_dir = os.path.join(os.path.dirname(__file__), '..', 'torchquantum', 'examples', 'quest')
if os.path.exists(quest_dir):
    sys.path.insert(0, quest_dir)
else:
    print(f"Error: Could not find quest directory at {quest_dir}")
    print("Make sure torchquantum submodule is initialized.")
    sys.exit(1)

# --- 1. Import QuEst Model & Utils ---
try:
    from core.datasets.model import Simple_Model
    from utils.circ_dag_converter import circ_to_dag_with_data, build_my_noise_dict
    from torchpack.utils.config import configs
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Usage: Run this script from the LayerDAG root directory")
    print("Make sure torchquantum submodule is initialized: git submodule update --init")
    sys.exit(1)

# --- 2. Define Circuits in Qiskit ---
# Circuit 1: Simple Bell State (High Fidelity)
qc1 = QuantumCircuit(2)
qc1.h(0)
qc1.cx(0, 1)
# Note: We don't add measurements as they're not needed for fidelity estimation
# and the converter doesn't handle measurement gates

# Circuit 2: Deep/Noisy Circuit (Low Fidelity)
qc2 = QuantumCircuit(2)
qc2.h(0)
for _ in range(5):  # Add depth to increase noise
    qc2.cx(0, 1)
    qc2.rx(0.5, 0)
# Note: We don't add measurements as they're not needed for fidelity estimation

# --- 3. Set up Noise Model ---
# Use FakeJakarta backend for noise model
backend = FakeJakarta()
props = backend.properties().to_dict()
mydict = build_my_noise_dict(props)

# Transpile circuits to match backend basis gates
qc1_transpiled = transpile(qc1, backend)
qc2_transpiled = transpile(qc2, backend)

# Remove measurement gates explicitly (they're not needed for fidelity estimation)
# and the converter doesn't handle them
from qiskit.converters import circuit_to_dag, dag_to_circuit

def remove_measurements(circuit):
    """Remove all measurement gates from a circuit."""
    dag = circuit_to_dag(circuit)
    nodes_to_remove = []
    for node in dag.op_nodes():
        if node.op.name == 'measure':
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        dag.remove_op_node(node)
    return dag_to_circuit(dag)

qc1_transpiled = remove_measurements(qc1_transpiled)
qc2_transpiled = remove_measurements(qc2_transpiled)

# --- 4. Convert to Graph Data ---
print("Converting circuits to graph data...")
data1 = circ_to_dag_with_data(qc1_transpiled, mydict)
data2 = circ_to_dag_with_data(qc2_transpiled, mydict)

# Add batch information (required for PyTorch Geometric)
from torch_geometric.data import Batch
data1.batch = torch.zeros(data1.x.size(0), dtype=torch.long)
data2.batch = torch.zeros(data2.x.size(0), dtype=torch.long)

print(f"Circuit 1: {data1.x.size(0)} nodes, {data1.edge_index.size(1)} edges")
print(f"Circuit 2: {data2.x.size(0)} nodes, {data2.edge_index.size(1)} edges")

# --- 5. Load Model Configuration & Initialize Model ---
# Load config from the huge/default experiment
config_path = os.path.join(quest_dir, 'exp', 'huge', 'default', 'config.yaml')
if os.path.exists(config_path):
    configs.load(config_path, recursive=True)
    print(f"Loaded config from {config_path}")
else:
    print(f"Warning: Config file not found at {config_path}")
    print("Using default model configuration...")
    # Set default config values
    class DefaultConfig:
        class model:
            name = "simple"
            use_only_global = False
            use_global_features = True
            use_gate_type = True
            use_qubit_index = True
            use_T1T2 = True
            use_gate_error = True
            use_gate_index = True
            num_layers = 2
    configs.model = DefaultConfig.model

# Initialize model
model = Simple_Model(configs.model)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

# --- 6. Load Checkpoint ---
checkpoint_path = os.path.join(quest_dir, 'exp', 'huge', 'default', 'model.pth')
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully")
else:
    print(f"Warning: No checkpoint found at {checkpoint_path}")
    print("Using random weights (predictions will be meaningless).")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# --- 7. Calculate Fidelity ---
with torch.no_grad():
    data1 = data1.to(device)
    data2 = data2.to(device)
    
    fid1 = model(data1)
    fid2 = model(data2)

# Get raw predictions
fid1_raw = fid1.item()
fid2_raw = fid2.item()

# Normalize to 0-1 range (assuming model outputs can be negative or > 1)
# Using sigmoid to ensure values are in [0, 1] range
fid1_normalized = torch.sigmoid(torch.tensor(fid1_raw)).item()
fid2_normalized = torch.sigmoid(torch.tensor(fid2_raw)).item()

# Alternative: Clamp to [0, 1] if values should be in that range
fid1_clamped = max(0.0, min(1.0, fid1_raw))
fid2_clamped = max(0.0, min(1.0, fid2_raw))

print(f"\n{'='*60}")
print(f"PyTorch Quantum Model Fidelity Predictions")
print(f"{'='*60}")
print(f"\nCircuit 1 (Bell State - Simple, High Fidelity Expected):")
print(f"  Raw Model Output:     {fid1_raw:.6f}")
print(f"  Sigmoid Normalized:   {fid1_normalized:.6f} (0-1 range)")
print(f"  Clamped to [0,1]:    {fid1_clamped:.6f}")
print(f"\nCircuit 2 (Deep/Noisy - Complex, Lower Fidelity Expected):")
print(f"  Raw Model Output:     {fid2_raw:.6f}")
print(f"  Sigmoid Normalized:   {fid2_normalized:.6f} (0-1 range)")
print(f"  Clamped to [0,1]:    {fid2_clamped:.6f}")
print(f"\n{'='*60}")
print(f"Note: The model predicts raw values. Higher values indicate higher fidelity.")
print(f"      Normalized values (sigmoid) ensure 0-1 range for interpretation.")
print(f"{'='*60}")