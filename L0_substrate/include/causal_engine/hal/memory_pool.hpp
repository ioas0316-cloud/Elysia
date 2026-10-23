#ifndef CAUSAL_ENGINE_HAL_MEMORY_POOL_HPP
#define CAUSAL_ENGINE_HAL_MEMORY_POOL_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace causal_engine {
namespace hal {

// =========================================================================
// 1. SoA (Structure of Arrays) Memory Pool (Cache-Aligned)
// =========================================================================
template <size_t AlignBytes = 64>
class SoAMemoryPool {
private:
    size_t capacity_ = 0;
    size_t active_count_ = 0;

    // Parallel attribute arrays for SIMD coalesced access
    float* pos_x_ = nullptr;
    float* pos_y_ = nullptr;
    float* pos_z_ = nullptr;
    float* state_val_ = nullptr;
    uint32_t* node_ids_ = nullptr;
    uint8_t* manifested_flags_ = nullptr;

public:
    explicit SoAMemoryPool(size_t capacity) : capacity_(capacity) {
        allocate_aligned_memory();
    }

    ~SoAMemoryPool() {
        free_aligned_memory();
    }

    // Disable copy
    SoAMemoryPool(const SoAMemoryPool&) = delete;
    SoAMemoryPool& operator=(const SoAMemoryPool&) = delete;

    // Enable move
    SoAMemoryPool(SoAMemoryPool&& other) noexcept
        : capacity_(other.capacity_), active_count_(other.active_count_),
          pos_x_(other.pos_x_), pos_y_(other.pos_y_), pos_z_(other.pos_z_),
          state_val_(other.state_val_), node_ids_(other.node_ids_),
          manifested_flags_(other.manifested_flags_) {
        other.pos_x_ = nullptr;
        other.pos_y_ = nullptr;
        other.pos_z_ = nullptr;
        other.state_val_ = nullptr;
        other.node_ids_ = nullptr;
        other.manifested_flags_ = nullptr;
        other.capacity_ = 0;
        other.active_count_ = 0;
    }

    void allocate_aligned_memory() {
        size_t float_bytes = capacity_ * sizeof(float);
        size_t id_bytes = capacity_ * sizeof(uint32_t);
        size_t flag_bytes = capacity_ * sizeof(uint8_t);

        pos_x_ = static_cast<float*>(aligned_alloc_block(AlignBytes, float_bytes));
        pos_y_ = static_cast<float*>(aligned_alloc_block(AlignBytes, float_bytes));
        pos_z_ = static_cast<float*>(aligned_alloc_block(AlignBytes, float_bytes));
        state_val_ = static_cast<float*>(aligned_alloc_block(AlignBytes, float_bytes));
        node_ids_ = static_cast<uint32_t*>(aligned_alloc_block(AlignBytes, id_bytes));
        manifested_flags_ = static_cast<uint8_t*>(aligned_alloc_block(AlignBytes, flag_bytes));

        std::memset(state_val_, 0, float_bytes);
        std::memset(manifested_flags_, 0, flag_bytes);
    }

    void free_aligned_memory() {
        if (pos_x_) aligned_free_block(pos_x_);
        if (pos_y_) aligned_free_block(pos_y_);
        if (pos_z_) aligned_free_block(pos_z_);
        if (state_val_) aligned_free_block(state_val_);
        if (node_ids_) aligned_free_block(node_ids_);
        if (manifested_flags_) aligned_free_block(manifested_flags_);
    }

    bool add_element(uint32_t node_id, float x, float y, float z, float initial_state = 0.0f) {
        if (active_count_ >= capacity_) return false;
        size_t idx = active_count_++;
        node_ids_[idx] = node_id;
        pos_x_[idx] = x;
        pos_y_[idx] = y;
        pos_z_[idx] = z;
        state_val_[idx] = initial_state;
        manifested_flags_[idx] = 0;
        return true;
    }

    size_t size() const { return active_count_; }
    size_t capacity() const { return capacity_; }

    float* get_pos_x() { return pos_x_; }
    float* get_pos_y() { return pos_y_; }
    float* get_pos_z() { return pos_z_; }
    float* get_state_val() { return state_val_; }
    uint32_t* get_node_ids() { return node_ids_; }
    uint8_t* get_manifested_flags() { return manifested_flags_; }

private:
    void* aligned_alloc_block(size_t alignment, size_t size) {
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(size, alignment);
#else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            throw std::bad_alloc();
        }
#endif
        return ptr;
    }

    void aligned_free_block(void* ptr) {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

// =========================================================================
// 2. Zero-Copy Opaque Buffer Interop Handle (CUDA/Vulkan Shared VRAM)
// =========================================================================
struct ZeroCopyHandle {
    uint64_t opaque_handle_id = 0;
    void* shared_host_ptr = nullptr;
    size_t buffer_size_bytes = 0;
    bool timeline_semaphore_signaled = false;
    uint64_t semaphore_value = 0;
};

class ZeroCopyInteropBoundary {
private:
    std::vector<ZeroCopyHandle> handles_;

public:
    ZeroCopyHandle export_vulkan_handle_to_cuda(void* host_ptr, size_t bytes) {
        ZeroCopyHandle h;
        h.opaque_handle_id = reinterpret_cast<uint64_t>(host_ptr);
        h.shared_host_ptr = host_ptr;
        h.buffer_size_bytes = bytes;
        h.timeline_semaphore_signaled = false;
        h.semaphore_value = 0;
        handles_.push_back(h);
        return h;
    }

    void signal_timeline_semaphore(uint64_t handle_id, uint64_t val) {
        for (auto& h : handles_) {
            if (h.opaque_handle_id == handle_id) {
                h.timeline_semaphore_signaled = true;
                h.semaphore_value = val;
                break;
            }
        }
    }
};

} // namespace hal
} // namespace causal_engine

#endif // CAUSAL_ENGINE_HAL_MEMORY_POOL_HPP
