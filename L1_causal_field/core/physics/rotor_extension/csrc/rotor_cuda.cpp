#include <torch/extension.h>
#include <vector>

// Forward declarations of launch helper functions in .cu file
torch::Tensor rotor_sandwich_cuda_forward(
    torch::Tensor v, torch::Tensor u, torch::Tensor w, torch::Tensor theta);

std::vector<torch::Tensor> rotor_sandwich_cuda_backward(
    torch::Tensor g, torch::Tensor v, torch::Tensor u, torch::Tensor w, torch::Tensor theta);

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rotor_sandwich_cuda_forward, "Rotor Sandwich Forward (CUDA)");
    m.def("backward", &rotor_sandwich_cuda_backward, "Rotor Sandwich Backward (CUDA)");
}
