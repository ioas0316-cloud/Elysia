#ifndef NEXUS_IN_ENGINE_AI_H
#define NEXUS_IN_ENGINE_AI_H

#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>
#include "nexus_betti.h"

namespace CausalNexus {

enum class FenceStatus {
    SIGNALED,
    UNSIGNALED,
    TOPOLOGICAL_ROLLBACK
};

struct TextureRHIHandle {
    uint32_t width;
    uint32_t height;
    uint32_t format;
    uint64_t native_d3d12_resource_ptr;
    std::vector<uint8_t> device_memory_buffer;
};

class PingPongSwapchain {
private:
    TextureRHIHandle buffer_A;
    TextureRHIHandle buffer_B;
    int active_write_index;

public:
    PingPongSwapchain(uint32_t w, uint32_t h) : active_write_index(0) {
        buffer_A = {w, h, 0x1, 0xA000, std::vector<uint8_t>(w * h * 4, 0)};
        buffer_B = {w, h, 0x1, 0xB000, std::vector<uint8_t>(w * h * 4, 0)};
    }

    TextureRHIHandle& GetWriteBuffer() {
        return (active_write_index == 0) ? buffer_A : buffer_B;
    }

    TextureRHIHandle& GetDisplayBuffer() {
        return (active_write_index == 0) ? buffer_B : buffer_A;
    }

    void Swap() {
        active_write_index = 1 - active_write_index;
    }
};

class InEngineCausalAIBridge {
private:
    PingPongSwapchain swapchain;
    uint64_t current_fence_value;
    bool is_initialized;

public:
    InEngineCausalAIBridge(uint32_t width, uint32_t height)
        : swapchain(width, height), current_fence_value(0), is_initialized(true) {}

    FenceStatus ExecuteCausalInferenceAndBindDX12(
        const std::vector<uint8_t>& causal_bitmask,
        uint32_t height,
        uint32_t width,
        const BettiNumbers& expected_betti
    ) {
        if (!is_initialized) return FenceStatus::UNSIGNALED;

        // 1. Evaluate Betti numbers on the causal bitmask for topological integrity
        BettiNumbers current_betti = CalculateBetti2D(causal_bitmask, height, width);

        // 2. Hardware L1 Cache Direct Allocation Topological Rollback check
        if (current_betti.betti_0 != expected_betti.betti_0 ||
            current_betti.betti_1 != expected_betti.betti_1) {
            // Topological Disruption Detected: O(1) Rollback to previous display buffer
            return FenceStatus::TOPOLOGICAL_ROLLBACK;
        }

        // 3. Write inferred frame directly to active DX12 write buffer (Zero-Copy metaphor)
        TextureRHIHandle& write_buf = swapchain.GetWriteBuffer();
        uint32_t num_pixels = std::min<size_t>(write_buf.device_memory_buffer.size() / 4, causal_bitmask.size());
        for (size_t i = 0; i < num_pixels; ++i) {
            uint8_t bit = causal_bitmask[i] ? 255 : 0;
            write_buf.device_memory_buffer[i * 4 + 0] = bit; // Red (Trajectory)
            write_buf.device_memory_buffer[i * 4 + 1] = bit; // Green (Hitbox)
            write_buf.device_memory_buffer[i * 4 + 2] = bit; // Blue (Domain Lock)
            write_buf.device_memory_buffer[i * 4 + 3] = 255; // Alpha
        }

        // 4. Fence Signal and Swap Ping-Pong Buffers
        current_fence_value++;
        swapchain.Swap();
        return FenceStatus::SIGNALED;
    }

    TextureRHIHandle& GetCurrentDisplayBuffer() {
        return swapchain.GetDisplayBuffer();
    }

    uint64_t GetFenceValue() const { return current_fence_value; }
};

} // namespace CausalNexus

#endif // NEXUS_IN_ENGINE_AI_H
