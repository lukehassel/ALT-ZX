#include <cuda_runtime.h>
#include <cuComplex.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Helper Traits for Complex Types
template<typename T> struct CudaComplexType;
template<> struct CudaComplexType<c10::complex<float>> { using type = cuFloatComplex; };
template<> struct CudaComplexType<c10::complex<double>> { using type = cuDoubleComplex; };

// Helper Functions for Complex Arithmetic (Overloaded)
__device__ __forceinline__ cuFloatComplex cx_make(float r, float i) { return make_cuFloatComplex(r, i); }
__device__ __forceinline__ cuDoubleComplex cx_make(double r, double i) { return make_cuDoubleComplex(r, i); }

__device__ __forceinline__ cuFloatComplex cx_mul(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ __forceinline__ cuDoubleComplex cx_mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

__device__ __forceinline__ cuFloatComplex cx_add(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
__device__ __forceinline__ cuDoubleComplex cx_add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }

__device__ __forceinline__ float cx_abs(cuFloatComplex a) { return cuCabsf(a); }
__device__ __forceinline__ double cx_abs(cuDoubleComplex a) { return cuCabs(a); }

// Templated Kernel
template <typename scalar_t>
__global__ void mpo_update_kernel(
    const typename CudaComplexType<scalar_t>::type* __restrict__ M1, 
    const typename CudaComplexType<scalar_t>::type* __restrict__ M2, 
    const typename CudaComplexType<scalar_t>::type* __restrict__ G1, 
    const typename CudaComplexType<scalar_t>::type* __restrict__ G2, 
    typename CudaComplexType<scalar_t>::type* __restrict__ theta,    
    int dL, int dM, int dR
) {
    using cuComplexT = typename CudaComplexType<scalar_t>::type;
    using real_t = typename scalar_t::value_type;
    
    // blockIdx.z is the batch index
    int batch_idx = blockIdx.z;
    
    // Offset pointers for this batch
    // M1 stride: 2*2*dL*dM = 4*dL*dM
    const cuComplexT* M1_b = M1 + batch_idx * (4 * dL * dM);
    // M2 stride: 2*2*dM*dR = 4*dM*dR
    const cuComplexT* M2_b = M2 + batch_idx * (4 * dM * dR);
    // G1/G2 stride: 16
    const cuComplexT* G1_b = G1 + batch_idx * 16;
    const cuComplexT* G2_b = G2 + batch_idx * 16;
    // theta stride: 2*2*dL*2*2*dR = 16*dL*dR
    cuComplexT* theta_b = theta + batch_idx * (16 * dL * dR);

    // Output dimensions: [a, e, c, b, f, g]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_per_batch = 16 * dL * dR;
    
    if (idx >= total_elements_per_batch) return;
    
    // Backcompute indices from idx
    int remaining = idx;
    int g = remaining % dR; remaining /= dR;
    int f = remaining % 2; remaining /= 2;
    int b = remaining % 2; remaining /= 2;
    int c = remaining % dL; remaining /= dL;
    int e = remaining % 2; remaining /= 2;
    int a = remaining % 2;
    
    cuComplexT val = cx_make((real_t)0, (real_t)0);
    
    // G1 and G2 are 4x4. ae, hi, bf, jk are flattened indices.
    int ae = a * 2 + e;
    int bf = b * 2 + f;
    
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 2; ++i) {
            int hi = h * 2 + i;
            cuComplexT g1_val = G1_b[ae * 4 + hi];
            if (cx_abs(g1_val) < (real_t)1e-12) continue;
            
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    int jk = j * 2 + k;
                    cuComplexT g2_val = G2_b[bf * 4 + jk];
                    if (cx_abs(g2_val) < (real_t)1e-12) continue;
                    
                    cuComplexT g1g2 = cx_mul(g1_val, g2_val);
                    
                    for (int d = 0; d < dM; ++d) {
                        // M1 shape (2, 2, dL, dM) index (h, j, c, d)
                        int m1_idx = (((h * 2 + j) * dL + c) * dM + d);
                        cuComplexT m1_val = M1_b[m1_idx];
                        
                        // M2 shape (2, 2, dM, dR) index (i, k, d, g)
                        int m2_idx = (((i * 2 + k) * dM + d) * dR + g);
                        cuComplexT m2_val = M2_b[m2_idx];
                        
                        val = cx_add(val, cx_mul(g1g2, cx_mul(m1_val, m2_val)));
                    }
                }
            }
        }
    }
    
    theta_b[idx] = val;
}

torch::Tensor mpo_update_cuda(
    torch::Tensor M1,
    torch::Tensor M2,
    torch::Tensor G1,
    torch::Tensor G2
) {
    at::cuda::CUDAGuard device_guard(M1.device());
    
    // Check if batched (5D) or single (4D)
    int B, dL, dM, dR;
    if (M1.dim() == 5) {
        B = M1.size(0);
        dL = M1.size(3);
        dM = M1.size(4);
        dR = M2.size(4); 
    } else {
        B = 1;
        dL = M1.size(2);
        dM = M1.size(3);
        dR = M2.size(3);
    }
    
    auto options = torch::TensorOptions().dtype(M1.dtype()).device(M1.device());
    torch::Tensor theta;
    if (M1.dim() == 5) {
        theta = torch::empty({B, 2, 2, dL, 2, 2, dR}, options);
    } else {
        theta = torch::empty({2, 2, dL, 2, 2, dR}, options);
    }
    
    int total_elements_per_batch = 16 * dL * dR;
    int threads = 256;
    int blocks_x = (total_elements_per_batch + threads - 1) / threads;
    dim3 grid(blocks_x, 1, B);
    
    AT_DISPATCH_COMPLEX_TYPES(M1.scalar_type(), "mpo_update_cuda", ([&] {
        using cuComplexT = typename CudaComplexType<scalar_t>::type;
        
        mpo_update_kernel<scalar_t><<<grid, threads>>>(
            (const cuComplexT*)M1.data_ptr<scalar_t>(),
            (const cuComplexT*)M2.data_ptr<scalar_t>(),
            (const cuComplexT*)G1.data_ptr<scalar_t>(),
            (const cuComplexT*)G2.data_ptr<scalar_t>(),
            (cuComplexT*)theta.data_ptr<scalar_t>(),
            dL, dM, dR
        );
    }));
    
    return theta;
}
