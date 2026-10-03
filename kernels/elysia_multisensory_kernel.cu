#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// Fused Multi-Sensory Resonance CUDA Kernel
// Audio (k_audio, omega_audio) + IMU Acceleration (imu_accel) + Text Latent (z_text)
// ============================================================================
__global__ void multisensory_sdf_resonance_kernel(
    const float* __restrict__ query_x,          // [Batch, N_points, 64]
    const float* __restrict__ k_audio,          // [Batch, 64] (Audio Wavevector)
    const float* __restrict__ omega_audio,      // [Batch] (Audio Angular Frequency)
    const float* __restrict__ amp_audio,        // [Batch, 1]
    const float* __restrict__ imu_accel,        // [Batch, 3] (IMU Accel: [ax, ay, az])
    const float* __restrict__ z_text,           // [Batch, 64] (Text Latent Phase)
    float* __restrict__ d_out,                  // Output Distance: [Batch, N_points, 1]
    const int N_points,
    const int Dim,                              // Dim = 64
    const float t_curr,
    const float sigma
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (point_idx >= N_points) return;

    int x_offset = (batch_idx * N_points + point_idx) * Dim;
    int out_offset = batch_idx * N_points + point_idx;

    // 1. IMU Acceleration Spatial Gravity Warp (XYZ 3D Space)
    float ax = imu_accel[batch_idx * 3 + 0];
    float ay = imu_accel[batch_idx * 3 + 1];
    float az = imu_accel[batch_idx * 3 + 2];

    float x0 = query_x[x_offset + 0] + ax * 0.1f;
    float x1 = query_x[x_offset + 1] + ay * 0.1f;
    float x2 = query_x[x_offset + 2] + az * 0.1f;

    // 2. Base Distance & Multi-Sensory Phase Inner Products
    float norm_sq = x0 * x0 + x1 * x1 + x2 * x2;
    float k_dot_x = x0 * k_audio[batch_idx * Dim + 0] +
                    x1 * k_audio[batch_idx * Dim + 1] +
                    x2 * k_audio[batch_idx * Dim + 2];
    float z_dot_x = x0 * z_text[batch_idx * Dim + 0] +
                    x1 * z_text[batch_idx * Dim + 1] +
                    x2 * z_text[batch_idx * Dim + 2];

    #pragma unroll 8
    for (int d = 3; d < 64; ++d) { // D=3~63 Dimension Unrolling
        float x_val = query_x[x_offset + d];
        norm_sq += x_val * x_val;
        k_dot_x += x_val * k_audio[batch_idx * Dim + d];
        z_dot_x += x_val * z_text[batch_idx * Dim + d];
    }

    float norm_x = sqrtf(norm_sq + 1e-8f);
    float d_base = norm_x - 1.0f; // Base Sphere SDF

    // 3. Audio Phase + Text Phase High-dimensional Coupling
    float omega_t = fmodf(omega_audio[batch_idx] * t_curr, 6.28318530718f);
    float total_phase = k_dot_x + z_dot_x - omega_t;

    // 4. Decay Masking & Perturbation Addition
    float mask = expf(-fabsf(d_base) / sigma);
    float perturbation = amp_audio[batch_idx] * mask * cosf(total_phase);

    d_out[out_offset] = d_base + perturbation;
}

extern "C" void launch_multisensory_sdf_kernel_impl(
    const float* query_x,
    const float* k_audio,
    const float* omega_audio,
    const float* amp_audio,
    const float* imu_accel,
    const float* z_text,
    float* d_out,
    int batch_size,
    int n_points,
    int dim,
    float t_curr,
    float sigma,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    dim3 blocksPerGrid((n_points + threadsPerBlock - 1) / threadsPerBlock, batch_size);

    multisensory_sdf_resonance_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        query_x, k_audio, omega_audio, amp_audio, imu_accel, z_text, d_out,
        n_points, dim, t_curr, sigma
    );
}
