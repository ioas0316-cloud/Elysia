#ifndef FIBER_BUNDLE_GEODESIC_HPP
#define FIBER_BUNDLE_GEODESIC_HPP

#include <cuda_runtime.h>
#include <cstdint>

// 5 Primal Human Sensory Input Ports (Vision, Audition, Somatosensory, Olfaction, Gustation)
#define NUM_SENSORY_PORTS 5

// 4D Fiber Bundle point state (AOSOA aligned)
struct alignas(16) FiberPoint {
    float4 coords;    // (t, x1, x2, x3) - Temporal trajectory axis + 3D structural volume coordinates
    float4 velocity;  // (dt/dtau, dx1/dtau, dx2/dtau, dx3/dtau) - Causal geodesic flow velocity
};

// Fiber metric tensor & gauge potential
struct alignas(16) FiberMetric {
    float h[3][3];    // 3D fiber spatial metric h_ij (inter-sensory causal distance)
    float A_t[3];     // Temporal gauge connection A_t^i
};

// Sensory port occupancy weights (Vision, Audition, Somatosensory, Olfaction, Gustation)
struct alignas(16) SensoryPortBoundary {
    float weights[NUM_SENSORY_PORTS]; // Sensory wave impact proportion per port (sum = 1.0)
};

#ifdef __cplusplus
extern "C" {
#endif

// Launch CUDA kernel for Fiber Bundle Geodesic Flow integration
void launch_fiber_geodesic_flow(
    FiberPoint* d_points,
    const FiberMetric* d_metrics,
    const SensoryPortBoundary* d_boundaries,
    int num_points,
    float d_tau,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // FIBER_BUNDLE_GEODESIC_HPP
