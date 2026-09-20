#include "causal_engine/fiber_bundle_geodesic.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void compute_christoffel(
    const FiberMetric& metric,
    float gamma[4][4][4]
) {
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                gamma[m][a][b] = 0.0f;
            }
        }
    }

    // Reflect gauge potential A_t on temporal deformation (\Gamma^i_{00} components)
    #pragma unroll
    for (int i = 1; i <= 3; ++i) {
        gamma[i][0][0] = -0.5f * metric.A_t[i - 1];
        gamma[i][0][i] = 0.5f * metric.A_t[i - 1];
        gamma[i][i][0] = 0.5f * metric.A_t[i - 1];
    }
}

__global__ void fiber_geodesic_flow_kernel(
    FiberPoint* __restrict__ points,
    const FiberMetric* __restrict__ metrics,
    const SensoryPortBoundary* __restrict__ boundaries,
    int num_points,
    float d_tau
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    FiberPoint pt = points[idx];
    FiberMetric met = metrics[idx];

    float gamma[4][4][4];
    compute_christoffel(met, gamma);

    // 1. Geodesic acceleration d^2(z^\mu) / d\tau^2 = -\Gamma^\mu_{\alpha\beta} v^\alpha v^\beta
    float accel[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[4] = {pt.velocity.x, pt.velocity.y, pt.velocity.z, pt.velocity.w};

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        float sum = 0.0f;
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                sum += gamma[m][a][b] * v[a] * v[b];
            }
        }
        accel[m] = -sum;
    }

    // Ensure temporal velocity strictly monotonic (dt/dtau > 0)
    if (v[0] + accel[0] * d_tau <= 0.0f) {
        accel[0] = 0.0f;
    }

    // 2. Symplectic Euler integration
    pt.velocity.x += accel[0] * d_tau;
    pt.velocity.y += accel[1] * d_tau;
    pt.velocity.z += accel[2] * d_tau;
    pt.velocity.w += accel[3] * d_tau;

    pt.coords.x += pt.velocity.x * d_tau; // Temporal trajectory axis (t)
    pt.coords.y += pt.velocity.y * d_tau; // Fiber x1
    pt.coords.z += pt.velocity.z * d_tau; // Fiber x2
    pt.coords.w += pt.velocity.w * d_tau; // Fiber x3

    points[idx] = pt;
}

extern "C" void launch_fiber_geodesic_flow(
    FiberPoint* d_points,
    const FiberMetric* d_metrics,
    const SensoryPortBoundary* d_boundaries,
    int num_points,
    float d_tau,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
    fiber_geodesic_flow_kernel<<<grid_size, block_size, 0, stream>>>(
        d_points, d_metrics, d_boundaries, num_points, d_tau
    );
}
