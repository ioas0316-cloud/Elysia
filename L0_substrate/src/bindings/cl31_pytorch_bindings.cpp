#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA launcher
#ifdef ELYSIA_WITH_CUDA
extern "C" void launch_cl31_soa_weave_cuda(
    const float* d_A_s, const float* d_A_v0, const float* d_A_v1, const float* d_A_v2, const float* d_A_v3,
    const float* d_B_s, const float* d_B_v0, const float* d_B_v1, const float* d_B_v2, const float* d_B_v3,
    float* d_Out_s, float* d_Out_b0, float* d_Out_b1, float* d_Out_b2,
    float* d_Out_b3, float* d_Out_b4, float* d_Out_b5,
    int N);
#endif

// CPU reference launcher
#include <elysia/cl31_sta.hpp>

std::vector<torch::Tensor> cl31_weave_pytorch(
    torch::Tensor A_s, torch::Tensor A_v0, torch::Tensor A_v1, torch::Tensor A_v2, torch::Tensor A_v3,
    torch::Tensor B_s, torch::Tensor B_v0, torch::Tensor B_v1, torch::Tensor B_v2, torch::Tensor B_v3)
{
    // Ensure all input tensors are contiguous
    auto c_A_s  = A_s.contiguous();  auto c_A_v0 = A_v0.contiguous();
    auto c_A_v1 = A_v1.contiguous(); auto c_A_v2 = A_v2.contiguous(); auto c_A_v3 = A_v3.contiguous();

    auto c_B_s  = B_s.contiguous();  auto c_B_v0 = B_v0.contiguous();
    auto c_B_v1 = B_v1.contiguous(); auto c_B_v2 = B_v2.contiguous(); auto c_B_v3 = B_v3.contiguous();

    int N = c_A_s.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(c_A_s.device());

    torch::Tensor Out_s  = torch::zeros({N}, options);
    torch::Tensor Out_b0 = torch::zeros({N}, options);
    torch::Tensor Out_b1 = torch::zeros({N}, options);
    torch::Tensor Out_b2 = torch::zeros({N}, options);
    torch::Tensor Out_b3 = torch::zeros({N}, options);
    torch::Tensor Out_b4 = torch::zeros({N}, options);
    torch::Tensor Out_b5 = torch::zeros({N}, options);

    if (c_A_s.is_cuda()) {
#ifdef ELYSIA_WITH_CUDA
        launch_cl31_soa_weave_cuda(
            c_A_s.data_ptr<float>(), c_A_v0.data_ptr<float>(), c_A_v1.data_ptr<float>(), c_A_v2.data_ptr<float>(), c_A_v3.data_ptr<float>(),
            c_B_s.data_ptr<float>(), c_B_v0.data_ptr<float>(), c_B_v1.data_ptr<float>(), c_B_v2.data_ptr<float>(), c_B_v3.data_ptr<float>(),
            Out_s.data_ptr<float>(), Out_b0.data_ptr<float>(), Out_b1.data_ptr<float>(), Out_b2.data_ptr<float>(),
            Out_b3.data_ptr<float>(), Out_b4.data_ptr<float>(), Out_b5.data_ptr<float>(),
            N);
#else
        TORCH_CHECK(false, "Elysia engine was compiled without CUDA support");
#endif
    } else {
        elysia::cl31_weave_soa_cpu(
            c_A_s.data_ptr<float>(), c_A_v0.data_ptr<float>(), c_A_v1.data_ptr<float>(), c_A_v2.data_ptr<float>(), c_A_v3.data_ptr<float>(),
            c_B_s.data_ptr<float>(), c_B_v0.data_ptr<float>(), c_B_v1.data_ptr<float>(), c_B_v2.data_ptr<float>(), c_B_v3.data_ptr<float>(),
            Out_s.data_ptr<float>(), Out_b0.data_ptr<float>(), Out_b1.data_ptr<float>(), Out_b2.data_ptr<float>(),
            Out_b3.data_ptr<float>(), Out_b4.data_ptr<float>(), Out_b5.data_ptr<float>(),
            N);
    }

    return {Out_s, Out_b0, Out_b1, Out_b2, Out_b3, Out_b4, Out_b5};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cl31_weave", &cl31_weave_pytorch, "Cl(3,1) Geometric Product Weave Kernel");
}
