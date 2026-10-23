#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// Zero-Extra-Sample Analytic Gradient (Normal Vector) CUDA Kernel
// Calculates distance and 3D Normal Vector simultaneously with 0 extra SDF evaluations
// ============================================================================
__global__ void analytic_sdf_normal_kernel(
    const float* __restrict__ query_pos,    // [N_points, 3] (3D Spatial Positions)
    const float* __restrict__ k_ext,        // [3] (3D Spatial Wavevector)
    const float omega_t,                    // ω * t
    const float amplitude,
    const float sigma,
    float* __restrict__ d_out,              // Output Distance: [N_points, 1]
    float* __restrict__ normal_out,         // Output Normal Vector: [N_points, 3]
    const int N_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_points) return;

    // 1. 3D Position
    float x = query_pos[idx * 3 + 0];
    float y = query_pos[idx * 3 + 1];
    float z = query_pos[idx * 3 + 2];

    // 2. Base Sphere SDF & Normal Derivative
    float norm_x = sqrtf(x * x + y * y + z * z + 1e-8f);
    float d_base = norm_x - 1.0f;

    float grad_base_x = x / norm_x;
    float grad_base_y = y / norm_x;
    float grad_base_z = z / norm_x;

    // 3. Phase Calculation
    float k_x = k_ext[0], k_y = k_ext[1], k_z = k_ext[2];
    float phase = (k_x * x + k_y * y + k_z * z) - omega_t;

    // 4. Analytical Chain Rule Term Calculations
    float cos_p = cosf(phase);
    float sin_p = sinf(phase);
    float mask = expf(-fabsf(d_base) / sigma);

    d_out[idx] = d_base + amplitude * mask * cos_p;

    // 5. Analytical Gradient Assembly (∇d = ∇d_base + ∇Perturbation)
    float sgn_d = (d_base > 0.0f) ? 1.0f : ((d_base < 0.0f) ? -1.0f : 0.0f);
    float d_mask_term = -(sgn_d / sigma) * mask * cos_p;
    float d_phase_term = -mask * sin_p;

    float nx = grad_base_x * (1.0f + amplitude * d_mask_term) + k_x * (amplitude * d_phase_term);
    float ny = grad_base_y * (1.0f + amplitude * d_mask_term) + k_y * (amplitude * d_phase_term);
    float nz = grad_base_z * (1.0f + amplitude * d_mask_term) + k_z * (amplitude * d_phase_term);

    float n_len = sqrtf(nx * nx + ny * ny + nz * nz + 1e-8f);
    normal_out[idx * 3 + 0] = nx / n_len;
    normal_out[idx * 3 + 1] = ny / n_len;
    normal_out[idx * 3 + 2] = nz / n_len;
}

extern "C" void launch_analytic_normal_kernel_impl(
    const float* query_pos,
    const float* k_ext,
    float omega_t,
    float amplitude,
    float sigma,
    float* d_out,
    float* normal_out,
    int n_points,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_points + threadsPerBlock - 1) / threadsPerBlock;

    analytic_sdf_normal_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        query_pos, k_ext, omega_t, amplitude, sigma, d_out, normal_out, n_points
    );
}
