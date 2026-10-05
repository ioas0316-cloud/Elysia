#ifndef VRAM_MONITOR_HPP
#define VRAM_MONITOR_HPP

#include <string>
#include <stdexcept>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

struct VRAMState {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    float used_mb;
    float free_mb;
    float usage_ratio;
};

class VRAMTracker {
public:
    static VRAMState get_current_state() {
        size_t free_b = 0, total_b = 0;
        cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
        if (err != cudaSuccess) {
            // In CPU fallback mode or if CUDA query is mocked
            free_b = 2048ULL * 1024 * 1024;
            total_b = 3072ULL * 1024 * 1024;
        }

        VRAMState state;
        state.free_bytes  = free_b;
        state.total_bytes = total_b;
        state.used_bytes  = (total_b >= free_b) ? (total_b - free_b) : 0;
        state.used_mb     = static_cast<float>(state.used_bytes) / (1024.0f * 1024.0f);
        state.free_mb     = static_cast<float>(state.free_bytes) / (1024.0f * 1024.0f);
        state.usage_ratio = (total_b > 0) ? (static_cast<float>(state.used_bytes) / static_cast<float>(total_b)) : 0.0f;

        return state;
    }

    // Allows setting simulated VRAM usage for testing/benchmarking
    static VRAMState get_simulated_state(float simulated_used_mb, float total_mb = 3072.0f) {
        VRAMState state;
        state.total_bytes = static_cast<size_t>(total_mb * 1024.0f * 1024.0f);
        state.used_bytes = static_cast<size_t>(simulated_used_mb * 1024.0f * 1024.0f);
        if (state.used_bytes > state.total_bytes) state.used_bytes = state.total_bytes;
        state.free_bytes = state.total_bytes - state.used_bytes;
        state.used_mb = simulated_used_mb;
        state.free_mb = total_mb - simulated_used_mb;
        state.usage_ratio = state.used_mb / total_mb;
        return state;
    }
};

#endif // VRAM_MONITOR_HPP
