"""Debug script to verify gflow labels are correct."""
import torch
import sys
import pyzx as zx
sys.path.insert(0, '.')

print('=== Loading Dataset ===')
data_list = torch.load('GflowNet/dataset.pt', weights_only=False)
print(f'Total samples: {len(data_list)}')

# Get label distribution
gflow_1_samples = [d for d in data_list[:1000] if d.y.item() == 1.0]
gflow_0_samples = [d for d in data_list[:1000] if d.y.item() == 0.0]
print(f'In first 1000: gflow=1: {len(gflow_1_samples)}, gflow=0: {len(gflow_0_samples)}')

# Check if gflow property matches labels
# We need to reconstruct the PyZX graph from PyG data to check gflow
# But we don't have the original PyZX graph - we only have features

# Instead, let's check the dataset generation directly
print('\n=== Verifying Gflow Labels (Regenerating Samples) ===')

from GflowNet.dataset import create_random_circuit_graph, break_gflow, check_gflow
import random
import numpy as np

# Test: Generate some circuit graphs and check their gflow
print('\nTest 1: Fresh circuit graphs (should ALL have gflow)')
random.seed(42)
np.random.seed(42)
correct = 0
for i in range(20):
    graph = create_random_circuit_graph(5, 5, seed=42+i)
    has_gflow = check_gflow(graph)
    if has_gflow:
        correct += 1
    else:
        print(f'  Sample {i}: UNEXPECTED - circuit graph has no gflow!')
print(f'  Result: {correct}/20 circuit graphs have gflow')

# Test: Break gflow and verify it's broken
print('\nTest 2: Broken graphs (should have NO gflow)')
random.seed(42)
np.random.seed(42)
broken_count = 0
for i in range(20):
    graph = create_random_circuit_graph(5, 5, seed=42+i)
    broken = break_gflow(graph)
    has_gflow = check_gflow(broken)
    if not has_gflow:
        broken_count += 1
    else:
        print(f'  Sample {i}: break_gflow FAILED - graph still has gflow!')
print(f'  Result: {broken_count}/20 graphs successfully had gflow broken')

# Test: Check what check_gflow actually returns
print('\n=== Testing check_gflow function ===')
from pyzx.flow import gflow

graph = create_random_circuit_graph(3, 3, seed=123)
print(f'\nCircuit graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges')
print(f'  Inputs: {list(graph.inputs())}')
print(f'  Outputs: {list(graph.outputs())}')

try:
    result = gflow(graph)
    print(f'  gflow() returned: {type(result)}, is None: {result is None}')
    if result is not None:
        print(f'  gflow exists: YES')
except Exception as e:
    print(f'  gflow() error: {e}')

# Break and check
broken = break_gflow(graph)
print(f'\nBroken graph: {broken.num_vertices()} vertices, {broken.num_edges()} edges')
try:
    result = gflow(broken)
    print(f'  gflow() returned: {type(result)}, is None: {result is None}')
    if result is not None:
        print(f'  gflow exists: YES - BREAK FAILED!')
    else:
        print(f'  gflow exists: NO - BREAK SUCCEEDED!')
except Exception as e:
    print(f'  gflow() error: {e}')

print('\n=== Done ===')


