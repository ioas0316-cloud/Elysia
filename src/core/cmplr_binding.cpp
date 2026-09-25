#include <torch/extension.h>
#include "elysia/core/cmplr_kernel.cuh"

#ifdef WITH_CUDA
void cmplr_step_cuda(
    torch::Tensor Psi,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor K_tensors,
    float dt
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &cmplr_step_cuda, "CMPLR Single Step Relaxation (CUDA)");
}
#else
void cmplr_step_stub(
    torch::Tensor Psi,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor K_tensors,
    float dt
) {
    TORCH_CHECK(false, "CUDA support is not enabled in this build.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &cmplr_step_stub, "CMPLR Single Step Relaxation (CPU Stub)");
}
#endif
