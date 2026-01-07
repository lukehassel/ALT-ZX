# MPO: GPU-Accelerated Fidelity Calculation

Matrix Product Operator engine for computing ground-truth fidelity between quantum circuits.

## Features

- **GPU Acceleration**: Custom CUDA kernels for tensor contraction
- **Batch Processing**: Computes fidelity for circuit batches
- **PyTorch Integration**: Works with PyTorch tensors

## Files

| File | Description |
|------|-------------|
| `mpo_gpu.py` | Python interface for GPU operations |
| `mpo_cuda_kernel.cu` | CUDA tensor operations |
| `fidelity.py` | High-level fidelity API |
| `setup.py` | Build script |
