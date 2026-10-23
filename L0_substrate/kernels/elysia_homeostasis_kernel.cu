#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// CUDA Kernel: Homeostatic Energy Clamping & Eikonal Regularization
// Prevents Latent Space gradient explosion and preserves memory field structural bounds
// ============================================================================
__global__ void homeostasis_regularization_kernel(
    float* __restrict__ d_field,              // [Batch, N_points, 1] (SDF Field)
    const float* __restrict__ query_pos,      // [Batch, N_points, Dim]
    const float target_energy,
    const float current_energy,
    const float lambda_scaling,
    const float nu_diffusion,
    const float beta_eikonal,
    const int N_points,
    const int Dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx >= N_points) return;

    int offset = batch_idx * N_points + idx;
    int pos_offset = offset * Dim;

    float d_val = d_field[offset];

    // 1. Distance Gradient Approximation
    float norm_sq = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < 3; ++d) {
        float pos = query_pos[pos_offset + d];
        norm_sq += pos * pos;
    }
    float norm_x = sqrtf(norm_sq + 1e-8f);

    float grad_norm = fabsf(d_val) / (norm_x + 1e-8f);
    float eikonal_err = grad_norm - 1.0f;

    // 2. Global Synaptic Scaling Term
    float energy_diff = current_energy - target_energy;
    float scaling_force = -lambda_scaling * energy_diff * d_val;

    // 3. Eikonal Regularization Force
    float eikonal_force = -beta_eikonal * eikonal_err * d_val;

    // 4. Laplacian Smooth Diffusion
    float diffusion_force = -nu_diffusion * d_val * 0.01f;

    // 5. Update SDF Field with Homeostatic PDE Step
    float updated_d = d_val + (scaling_force + eikonal_force + diffusion_force);

    // Hard Bounds Guard
    d_field[offset] = fmaxf(-3.0f, fminf(3.0f, updated_d));
}

extern "C" void launch_homeostasis_kernel_impl(
    float* d_field,
    const float* query_pos,
    float target_energy,
    float current_energy,
    float lambda_scaling,
    float nu_diffusion,
    float beta_eikonal,
    int batch_size,
    int n_points,
    int dim,
    cudaStream_t stream
) {
    int threads = 256;
    dim3 blocks((n_points + threads - 1) / threads, batch_size);

    homeostasis_regularization_kernel<<<blocks, threads, 0, stream>>>(
        d_field, query_pos, target_energy, current_energy,
        lambda_scaling, nu_diffusion, beta_eikonal,
        n_points, dim
    );
}
