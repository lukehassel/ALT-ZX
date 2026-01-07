import torch
import time


def benchmark_svd_batching():
    device = torch.device('cuda')
    dtype = torch.complex64
    
    B = 10
    M, N = 512, 512
    rank = 256
    
    A_batch = torch.randn(B, M, N, dtype=dtype, device=device)
    
    torch.linalg.svd(A_batch[0], full_matrices=False)
    
    print(f"Benchmarking SVD on {B} matrices of size {M}x{N}...")
    
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(B):
        torch.linalg.svd(A_batch[i], full_matrices=False)
    torch.cuda.synchronize()
    t_serial = time.time() - t0
    print(f"Serial SVD Time: {t_serial:.4f}s")
    
    torch.cuda.synchronize()
    t0 = time.time()
    torch.linalg.svd(A_batch, full_matrices=False)
    torch.cuda.synchronize()
    t_batched = time.time() - t0
    print(f"Batched SVD Time: {t_batched:.4f}s")
    
    print(f"Speedup: {t_serial / t_batched:.2f}x")


if __name__ == "__main__":
    benchmark_svd_batching()
