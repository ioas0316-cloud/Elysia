/*
 * Predictive Resonance Phase-Gated C++/CUDA Kernel
 * Implements Layer 1 Warp/Block-Level Reduction Early Exit and Layer 2/3 CDP Volitional Engine Kernel.
 */

#include <torch/extension.h>
#include <cmath>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

// Layer 2/3: Volitional Child Kernel (CUDA Dynamic Parallelism)
__global__ void volitional_engine_kernel(
    const float* __restrict__ x_input,
    float* __restrict__ internal_state,
    float prediction_error,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float learning_rate = 0.01f * (prediction_error - 0.05f);
        float delta = x_input[idx] - internal_state[idx];
        atomicAdd(&internal_state[idx], learning_rate * delta);
    }
}

// Layer 1: Passive Resonance Block/Grid Reduction Gate Kernel
__global__ void passive_resonance_gate_kernel(
    const float* __restrict__ x_input,
    float* __restrict__ internal_state,
    float* __restrict__ output_error,
    int* __restrict__ gated_flag,
    float tau_min,
    int dim
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float diff_sq = 0.0f;
    for (int i = idx; i < dim; i += blockDim.x * gridDim.x) {
        float val = x_input[i] - internal_state[i];
        diff_sq += val * val;
    }

    // Shared memory reduction within block
    sdata[tid] = diff_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output_error, sdata[0]);
    }

    __syncthreads();

    // Block 0 thread 0 performs threshold check & early exit / CDP launch
    if (blockIdx.x == 0 && tid == 0) {
        float total_error = sqrtf(*output_error);
        *output_error = total_error;

        if (total_error < tau_min) {
            *gated_flag = 1; // Passive bypass
        } else {
            *gated_flag = 0; // Volitional active
            int threads = 256;
            int blocks = (dim + threads - 1) / threads;
            volitional_engine_kernel<<<blocks, threads>>>(x_input, internal_state, total_error, dim);
        }
    }
}
#endif

// PyTorch C++ Extension Interface (with CPU fallback)
std::tuple<at::Tensor, float, bool> predictive_resonance_cuda_forward(
    at::Tensor x_input,
    at::Tensor internal_state,
    double tau_min
) {
    TORCH_CHECK(x_input.is_contiguous(), "x_input must be contiguous");
    TORCH_CHECK(internal_state.is_contiguous(), "internal_state must be contiguous");

    int dim = x_input.numel();
    float err = 0.0f;
    bool passive_gated = true;

    if (x_input.is_cuda() && internal_state.is_cuda()) {
#ifdef __CUDACC__
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(x_input.device());
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(x_input.device());

        auto out_err_t = torch::zeros({1}, options_float);
        auto gated_flag_t = torch::zeros({1}, options_int);

        int threads = 256;
        int blocks = (dim + threads - 1) / threads;
        size_t smem_size = threads * sizeof(float);

        passive_resonance_gate_kernel<<<blocks, threads, smem_size>>>(
            x_input.data_ptr<float>(),
            internal_state.data_ptr<float>(),
            out_err_t.data_ptr<float>(),
            gated_flag_t.data_ptr<int>(),
            static_cast<float>(tau_min),
            dim
        );

        cudaDeviceSynchronize();

        err = out_err_t.item<float>();
        passive_gated = (gated_flag_t.item<int>() == 1);
#endif
    } else {
        // CPU Fallback logic
        auto diff = x_input - internal_state;
        err = torch::norm(diff, 2).item<float>();

        if (err < tau_min) {
            passive_gated = true;
        } else {
            passive_gated = false;
            float lr = 0.01f * (err - static_cast<float>(tau_min));
            internal_state.add_(diff * lr);
        }
    }

    return std::make_tuple(internal_state, err, passive_gated);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &predictive_resonance_cuda_forward, "Predictive Resonance Forward Gate (C++/CUDA)");
}
