#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device gradient computation function (Fast s, Slow W)
__device__ void compute_gradients(
    const float* s, const float* W, const float* stress_grad,
    float* grad_s, float* grad_w,
    int N, float tau_s, float tau_w, float k_elastic, float lambda_w, const float* W0)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Fast Dynamics Gradient: ds/dt = (W*s - s - k_elastic * stress_grad) / tau_s
    if (row < N && col == 0) {
        float Ws_i = 0.0f;
        for (int j = 0; j < N; ++j) {
            Ws_i += W[row * N + j] * s[j];
        }
        float g_inv_s = s[row]; // Identity/linear activation
        grad_s[row] = (Ws_i - g_inv_s - k_elastic * stress_grad[row]) / tau_s;
    }

    // 2. Slow Dynamics Gradient: dW/dt = (0.5 * s_i * s_j - lambda * (W - W0) + Torque) / tau_w
    if (row < N && col < N) {
        int idx = row * N + col;
        float hebbian = 0.5f * s[row] * s[col];
        float decay = lambda_w * (W[idx] - W0[idx]);
        float torque = -k_elastic * stress_grad[row] * s[col]; // Topological deformation torque proxy

        grad_w[idx] = (hebbian - decay + torque) / tau_w;
    }
}

// Stage gradient evaluation kernel
__global__ void compute_gradients_kernel(
    const float* s, const float* W, const float* stress_grad, const float* W0,
    float* grad_s, float* grad_w,
    int N, float tau_s, float tau_w, float k_elastic, float lambda_w)
{
    compute_gradients(s, W, stress_grad, grad_s, grad_w, N, tau_s, tau_w, k_elastic, lambda_w, W0);
}

// Intermediate state update kernel
__global__ void update_temp_kernel(
    const float* s, const float* W,
    const float* k_s, const float* k_w,
    float* temp_s, float* temp_w,
    int N, float dt_factor)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    if (row < N && col == 0) {
        temp_s[row] = s[row] + dt_factor * k_s[row];
    }
    if (row < N && col < N) {
        temp_w[idx] = W[idx] + dt_factor * k_w[idx];
    }
}

// Final weighted accumulation kernel
__global__ void accumulation_kernel(
    float* s, float* W,
    const float* k1_s, const float* k2_s, const float* k3_s, const float* k4_s,
    const float* k1_w, const float* k2_w, const float* k3_w, const float* k4_w,
    int N, float dt)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    if (row < N && col == 0) {
        s[row] += (dt / 6.0f) * (k1_s[row] + 2.0f * k2_s[row] + 2.0f * k3_s[row] + k4_s[row]);
    }
    if (row < N && col < N) {
        W[idx] += (dt / 6.0f) * (k1_w[idx] + 2.0f * k2_w[idx] + 2.0f * k3_w[idx] + k4_w[idx]);
    }
}

// C++ Launcher Function
void rk4_step_cuda(
    torch::Tensor s,
    torch::Tensor W,
    torch::Tensor stress_grad,
    torch::Tensor W0,
    float dt,
    float tau_s,
    float tau_w,
    float k_elastic,
    float lambda_w)
{
    TORCH_CHECK(s.is_cuda(), "s must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(stress_grad.is_cuda(), "stress_grad must be a CUDA tensor");
    TORCH_CHECK(W0.is_cuda(), "W0 must be a CUDA tensor");

    TORCH_CHECK(s.is_contiguous(), "s must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");

    int N = s.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(s.device());
    auto k1_s = torch::empty({N}, options);
    auto k2_s = torch::empty({N}, options);
    auto k3_s = torch::empty({N}, options);
    auto k4_s = torch::empty({N}, options);
    auto temp_s = torch::empty({N}, options);

    auto k1_w = torch::empty({N, N}, options);
    auto k2_w = torch::empty({N, N}, options);
    auto k3_w = torch::empty({N, N}, options);
    auto k4_w = torch::empty({N, N}, options);
    auto temp_w = torch::empty({N, N}, options);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // STAGE 1: k1
    compute_gradients_kernel<<<gridDim, blockDim>>>(
        s.data_ptr<float>(), W.data_ptr<float>(), stress_grad.data_ptr<float>(), W0.data_ptr<float>(),
        k1_s.data_ptr<float>(), k1_w.data_ptr<float>(), N, tau_s, tau_w, k_elastic, lambda_w);

    // STAGE 2: k2
    update_temp_kernel<<<gridDim, blockDim>>>(
        s.data_ptr<float>(), W.data_ptr<float>(),
        k1_s.data_ptr<float>(), k1_w.data_ptr<float>(),
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), N, 0.5f * dt);

    compute_gradients_kernel<<<gridDim, blockDim>>>(
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), stress_grad.data_ptr<float>(), W0.data_ptr<float>(),
        k2_s.data_ptr<float>(), k2_w.data_ptr<float>(), N, tau_s, tau_w, k_elastic, lambda_w);

    // STAGE 3: k3
    update_temp_kernel<<<gridDim, blockDim>>>(
        s.data_ptr<float>(), W.data_ptr<float>(),
        k2_s.data_ptr<float>(), k2_w.data_ptr<float>(),
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), N, 0.5f * dt);

    compute_gradients_kernel<<<gridDim, blockDim>>>(
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), stress_grad.data_ptr<float>(), W0.data_ptr<float>(),
        k3_s.data_ptr<float>(), k3_w.data_ptr<float>(), N, tau_s, tau_w, k_elastic, lambda_w);

    // STAGE 4: k4
    update_temp_kernel<<<gridDim, blockDim>>>(
        s.data_ptr<float>(), W.data_ptr<float>(),
        k3_s.data_ptr<float>(), k3_w.data_ptr<float>(),
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), N, dt);

    compute_gradients_kernel<<<gridDim, blockDim>>>(
        temp_s.data_ptr<float>(), temp_w.data_ptr<float>(), stress_grad.data_ptr<float>(), W0.data_ptr<float>(),
        k4_s.data_ptr<float>(), k4_w.data_ptr<float>(), N, tau_s, tau_w, k_elastic, lambda_w);

    // FINAL ACCUMULATION
    accumulation_kernel<<<gridDim, blockDim>>>(
        s.data_ptr<float>(), W.data_ptr<float>(),
        k1_s.data_ptr<float>(), k2_s.data_ptr<float>(), k3_s.data_ptr<float>(), k4_s.data_ptr<float>(),
        k1_w.data_ptr<float>(), k2_w.data_ptr<float>(), k3_w.data_ptr<float>(), k4_w.data_ptr<float>(),
        N, dt);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &rk4_step_cuda, "Two-timescale RK4 Relaxation Step (CUDA)");
}
