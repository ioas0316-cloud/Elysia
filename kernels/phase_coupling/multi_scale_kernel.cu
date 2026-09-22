#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define TILE_SIZE 256

// Forward Kernel: Multi-Scale Cross-Frequency Phase Coupling
__global__ void multi_scale_forward_kernel(
    float* __restrict__ slow_phase_out,
    float* __restrict__ fast_phase_out,
    const float* __restrict__ slow_phase_in,
    const float* __restrict__ fast_phase_in,
    const float* __restrict__ slow_omega,
    const float* __restrict__ fast_omega,
    const float* __restrict__ metric,
    int num_nodes, float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_slow[TILE_SIZE];
    __shared__ float s_fast[TILE_SIZE];

    float my_slow = (idx < num_nodes) ? slow_phase_in[idx] : 0.0f;
    float my_fast = (idx < num_nodes) ? fast_phase_in[idx] : 0.0f;

    float sum_slow = 0.0f;
    float sum_fast = 0.0f;

    for (int tile = 0; tile < (num_nodes + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tile_idx = tile * TILE_SIZE + threadIdx.x;
        s_slow[threadIdx.x] = (tile_idx < num_nodes) ? slow_phase_in[tile_idx] : 0.0f;
        s_fast[threadIdx.x] = (tile_idx < num_nodes) ? fast_phase_in[tile_idx] : 0.0f;
        __syncthreads();

        if (idx < num_nodes) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                int j = tile * TILE_SIZE + k;
                if (j < num_nodes) {
                    float dist = metric[idx * num_nodes + j];
                    float influence = expf(-dist);
                    sum_slow += influence * sinf(s_slow[k] - my_slow);
                    sum_fast += influence * sinf(s_fast[k] - my_fast);
                }
            }
        }
        __syncthreads();
    }

    if (idx >= num_nodes) return;

    float top_down = M_mod * cosf(my_slow);
    float bottom_up = alpha_fb * sinf(my_fast - my_slow);

    float d_slow = slow_omega[idx] + (K_slow * sum_slow) + bottom_up;
    float d_fast = fast_omega[idx] + top_down + (K_fast * sum_fast);

    constexpr float TWO_PI = 2.0f * CUDART_PI_F;
    slow_phase_out[idx] = fmodf(my_slow + d_slow * dt + TWO_PI, TWO_PI);
    fast_phase_out[idx] = fmodf(my_fast + d_fast * dt + TWO_PI, TWO_PI);
}

// Backward Kernel: Analytical Gradient Propagation
__global__ void multi_scale_backward_kernel(
    float* __restrict__ grad_slow_phase,
    float* __restrict__ grad_fast_phase,
    float* __restrict__ grad_metric,
    const float* __restrict__ grad_slow_out,
    const float* __restrict__ grad_fast_out,
    const float* __restrict__ slow_phase_in,
    const float* __restrict__ fast_phase_in,
    const float* __restrict__ metric,
    int num_nodes, float dt, float K_slow, float K_fast, float M_mod, float alpha_fb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_slow[TILE_SIZE];
    __shared__ float s_fast[TILE_SIZE];
    __shared__ float s_grad_slow[TILE_SIZE];
    __shared__ float s_grad_fast[TILE_SIZE];

    float my_slow = (idx < num_nodes) ? slow_phase_in[idx] : 0.0f;
    float my_fast = (idx < num_nodes) ? fast_phase_in[idx] : 0.0f;
    float g_slow_i = (idx < num_nodes) ? grad_slow_out[idx] : 0.0f;
    float g_fast_i = (idx < num_nodes) ? grad_fast_out[idx] : 0.0f;

    float d_theta_i = g_slow_i * (1.0f - dt * alpha_fb * cosf(my_fast - my_slow))
                    - g_fast_i * dt * M_mod * sinf(my_slow);
    float d_phi_i   = g_fast_i * 1.0f + g_slow_i * dt * alpha_fb * cosf(my_fast - my_slow);

    for (int tile = 0; tile < (num_nodes + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tile_idx = tile * TILE_SIZE + threadIdx.x;
        if (tile_idx < num_nodes) {
            s_slow[threadIdx.x] = slow_phase_in[tile_idx];
            s_fast[threadIdx.x] = fast_phase_in[tile_idx];
            s_grad_slow[threadIdx.x] = grad_slow_out[tile_idx];
            s_grad_fast[threadIdx.x] = grad_fast_out[tile_idx];
        } else {
            s_slow[threadIdx.x] = 0.0f; s_fast[threadIdx.x] = 0.0f;
            s_grad_slow[threadIdx.x] = 0.0f; s_grad_fast[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (idx < num_nodes) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                int j = tile * TILE_SIZE + k;
                if (j < num_nodes) {
                    float dist = metric[idx * num_nodes + j];
                    float influence = expf(-dist);
                    float d_slow_ij = s_slow[k] - my_slow;
                    float d_fast_ij = s_fast[k] - my_fast;

                    d_theta_i += dt * K_slow * influence * (s_grad_slow[k] * cosf(-d_slow_ij) - g_slow_i * cosf(d_slow_ij));
                    d_phi_i   += dt * K_fast * influence * (s_grad_fast[k] * cosf(-d_fast_ij) - g_fast_i * cosf(d_fast_ij));

                    float g_metric_ij = -dt * influence * (
                        g_slow_i * K_slow * sinf(d_slow_ij) + g_fast_i * K_fast * sinf(d_fast_ij)
                    );
                    grad_metric[idx * num_nodes + j] = g_metric_ij;
                }
            }
        }
        __syncthreads();
    }

    if (idx < num_nodes) {
        grad_slow_phase[idx] = d_theta_i;
        grad_fast_phase[idx] = d_phi_i;
    }
}
