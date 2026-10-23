#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void multi_scale_forward_kernel(
    float* slow_out, float* fast_out, const float* slow_in, const float* fast_in,
    const float* slow_omega, const float* fast_omega, const float* metric,
    int num_nodes, float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
);

void multi_scale_backward_kernel(
    float* grad_slow_phase, float* grad_fast_phase, float* grad_metric,
    const float* grad_slow_out, const float* grad_fast_out,
    const float* slow_phase_in, const float* fast_phase_in, const float* metric,
    int num_nodes, float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
);

void launch_multi_scale_forward_cuda(
    torch::Tensor slow_out, torch::Tensor fast_out,
    torch::Tensor slow_in, torch::Tensor fast_in,
    torch::Tensor slow_omega, torch::Tensor fast_omega,
    torch::Tensor metric, float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
) {
    int num_nodes = slow_in.size(0);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    multi_scale_forward_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        slow_out.data_ptr<float>(), fast_out.data_ptr<float>(),
        slow_in.data_ptr<float>(), fast_in.data_ptr<float>(),
        slow_omega.data_ptr<float>(), fast_omega.data_ptr<float>(),
        metric.data_ptr<float>(), num_nodes, dt, K_slow, K_fast, M_mod, alpha_fb
    );
}

void launch_multi_scale_backward_cuda(
    torch::Tensor grad_slow_in, torch::Tensor grad_fast_in, torch::Tensor grad_metric,
    torch::Tensor grad_slow_out, torch::Tensor grad_fast_out,
    torch::Tensor slow_in, torch::Tensor fast_in, torch::Tensor metric,
    float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
) {
    int num_nodes = slow_in.size(0);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    multi_scale_backward_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        grad_slow_in.data_ptr<float>(), grad_fast_in.data_ptr<float>(), grad_metric.data_ptr<float>(),
        grad_slow_out.data_ptr<float>(), grad_fast_out.data_ptr<float>(),
        slow_in.data_ptr<float>(), fast_in.data_ptr<float>(), metric.data_ptr<float>(),
        num_nodes, dt, K_slow, K_fast, M_mod, alpha_fb
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_multi_scale_forward_cuda, "Phase Coupling Forward CUDA");
    m.def("backward", &launch_multi_scale_backward_cuda, "Phase Coupling Backward CUDA");
}
