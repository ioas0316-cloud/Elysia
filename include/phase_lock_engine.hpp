#pragma once

#include <torch/torch.h>
#include <tuple>

#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace ElysiaEngine {

class PhaseLockEnginePipeline {
public:
    struct Config {
        float R_cut = 2.5f;
        float dt = 0.01f;
        float phi_solid = 5.0f;
        float phi_gas = 0.2f;
        float gamma_m = 0.1f;
        float alpha = 0.5f;
        float beta = 1.0f;
        float tau_c = 3.0f;
        float c0 = 1.0f;
        float lambda_c = 0.5f;
    };

    PhaseLockEnginePipeline();
    explicit PhaseLockEnginePipeline(const Config& config);
    ~PhaseLockEnginePipeline() = default;

    // Direct C++ Tensor Update Step
    std::tuple<at::Tensor, at::Tensor> step(
        const at::Tensor& X,        // [N, 3] Tensor (CUDA or CPU)
        const at::Tensor& V,        // [N, 3] Tensor (CUDA or CPU)
        const at::Tensor& row_ptr,  // [N+1] Tensor
        const at::Tensor& col_idx,  // [E] Tensor
        at::Tensor& C_edge,         // [E] Tensor (In/Out)
        at::Tensor& M_edge          // [E] Tensor (In/Out)
    );

    const Config& config() const { return config_; }

private:
    Config config_;
    void validate_inputs(
        const at::Tensor& X,
        const at::Tensor& V,
        const at::Tensor& row_ptr,
        const at::Tensor& col_idx,
        const at::Tensor& C_edge,
        const at::Tensor& M_edge
    );

    std::tuple<at::Tensor, at::Tensor> step_cpu(
        const at::Tensor& X,
        const at::Tensor& V,
        const at::Tensor& row_ptr,
        const at::Tensor& col_idx,
        at::Tensor& C_edge,
        at::Tensor& M_edge
    );
};

} // namespace ElysiaEngine
