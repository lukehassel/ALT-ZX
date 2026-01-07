import sys
sys.path.insert(0, "mpo")
import torch
from mpo.gpu.mpo_gpu import MPOTensorGPU, DTYPE

device = "cuda:1"

print("=== Testing trace calculation ===")

for n_qubits in [4, 6, 8]:
    mpo = MPOTensorGPU(n_qubits, device=device)
    
    print(f"\n--- {n_qubits} qubits ---")
    for i, t in enumerate(mpo.tensors):
        print(f"  Tensor {i}: shape={t.shape}, sum={t.sum().item():.4f}")
    
    trace = mpo.trace_efficient()
    expected = 2 ** n_qubits
    print(f"  Trace: {trace.item():.6f}, Expected: {expected}, Ratio: {trace.item()/expected:.6f}")
    
    full = mpo.tensors[0]
    for t in mpo.tensors[1:]:
        full = torch.einsum('...ij,jk...->...ik...', full, t)
    print(f"  Full contracted shape: {full.shape}")
