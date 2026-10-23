#ifndef ELYSIA_CORE_CMPLR_KERNEL_CUH
#define ELYSIA_CORE_CMPLR_KERNEL_CUH

#include <torch/extension.h>

void cmplr_step_cuda(
    torch::Tensor Psi,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor K_tensors,
    float dt
);

#endif // ELYSIA_CORE_CMPLR_KERNEL_CUH
