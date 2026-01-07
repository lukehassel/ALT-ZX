#include <torch/extension.h>

// Forward declaration of the CUDA function
torch::Tensor mpo_update_cuda(
    torch::Tensor M1,
    torch::Tensor M2,
    torch::Tensor G1,
    torch::Tensor G2
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_theta", &mpo_update_cuda, "MPO update fused kernel (CUDA)");
}
