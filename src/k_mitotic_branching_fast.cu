#include "LockFreeCausalMemoryPool.cuh"

extern "C" __global__ void k_mitotic_branching_fast(
    CausalMemoryPoolDeviceHandle* pool,
    const int*  active_node_indices,
    const float curvature_threshold,
    const int   num_active_nodes
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_active_nodes) return;

    int parent_idx = active_node_indices[thread_idx];
    float cur = pool->curvature_pool[parent_idx];

    if (cur >= curvature_threshold) {
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        int child1_idx = gpu_causal_node_alloc(pool);
        int child2_idx = gpu_causal_node_alloc(pool);
#else
        int child1_idx = -1;
        int child2_idx = -1;
#endif

        if (child1_idx != -1 && child2_idx != -1) {
            float2 p = pool->rotor_pool[parent_idx];
            const float norm_factor = 0.70710678f;

            pool->rotor_pool[child1_idx] = make_float2(
                (p.x * 0.7071f - p.y * 0.7071f) * norm_factor,
                (p.x * 0.7071f + p.y * 0.7071f) * norm_factor
            );
            pool->rotor_pool[child2_idx] = make_float2(
                (p.x * 0.7071f + p.y * 0.7071f) * norm_factor,
                (-p.x * 0.7071f + p.y * 0.7071f) * norm_factor
            );

            pool->curvature_pool[child1_idx] = 0.0f;
            pool->curvature_pool[child2_idx] = 0.0f;
        }
    }
}
