#pragma once

#include "causal_engine/core/arena_allocator.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>
#include <stdexcept>
#include <functional>
#include <vector>

namespace causal_engine {

/**
 * @brief Stream Frame Header containing metadata for zero-copy stream packets.
 */
struct alignas(64) StreamFrameHeader {
    uint64_t frame_index{0};
    uint64_t timestamp_ns{0};
    uint32_t payload_bytes{0};
    uint32_t channel_id{0};
    uint32_t flags{0};
    uint32_t checksum{0};
};

/**
 * @brief Zero-Copy Interceptor Buffer Slot within the pre-allocated arena ring buffer.
 */
struct BufferSlot {
    StreamFrameHeader* header{nullptr};
    uint8_t* payload{nullptr};
    std::atomic<bool> is_ready{false};
};

/**
 * @brief High-Performance Zero-Copy Interceptor.
 *
 * Intercepts data flows directly in pre-allocated Arena memory without dynamic heap
 * allocations or redundant OS copy operations.
 */
class ZeroCopyInterceptor {
public:
    /**
     * @param arena Reference to the parent ArenaAllocator slab.
     * @param slot_count Number of pre-allocated ring-buffer slots.
     * @param max_payload_bytes Maximum payload capacity per slot.
     */
    ZeroCopyInterceptor(ArenaAllocator& arena, size_t slot_count = 16, size_t max_payload_bytes = 1024 * 1024)
        : m_arena(arena), m_slot_count(slot_count), m_max_payload_bytes(max_payload_bytes),
          m_write_index(0), m_read_index(0), m_dropped_frames(0) {
        if (slot_count == 0) {
            throw std::invalid_argument("Slot count must be greater than 0");
        }

        m_slots = arena.allocate_typed<BufferSlot>(m_slot_count);
        for (size_t i = 0; i < m_slot_count; ++i) {
            new (&m_slots[i]) BufferSlot();
            m_slots[i].header = arena.allocate_typed<StreamFrameHeader>(1);
            m_slots[i].payload = arena.allocate_typed<uint8_t>(m_max_payload_bytes);
            m_slots[i].is_ready.store(false, std::memory_order_relaxed);
        }
    }

    ~ZeroCopyInterceptor() = default;

    // Disable copy
    ZeroCopyInterceptor(const ZeroCopyInterceptor&) = delete;
    ZeroCopyInterceptor& operator=(const ZeroCopyInterceptor&) = delete;

    /**
     * @brief Acquire a write pointer to a buffer slot directly in arena memory.
     * Zero-copy allocation: Returns raw memory pointer without copies.
     */
    bool acquire_write_buffer(uint8_t** out_payload_ptr, StreamFrameHeader** out_header_ptr) {
        size_t current_w = m_write_index.load(std::memory_order_relaxed);
        size_t slot_idx = current_w % m_slot_count;
        BufferSlot& slot = m_slots[slot_idx];

        if (slot.is_ready.load(std::memory_order_acquire)) {
            // Buffer full, drop frame
            m_dropped_frames.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        *out_header_ptr = slot.header;
        *out_payload_ptr = slot.payload;
        return true;
    }

    /**
     * @brief Commit the written buffer slot to make it available for zero-copy consumption.
     */
    void commit_write_buffer(size_t payload_bytes, uint64_t frame_index = 0, uint32_t channel_id = 0) {
        size_t current_w = m_write_index.load(std::memory_order_relaxed);
        size_t slot_idx = current_w % m_slot_count;
        BufferSlot& slot = m_slots[slot_idx];

        slot.header->payload_bytes = static_cast<uint32_t>(payload_bytes);
        slot.header->frame_index = frame_index;
        slot.header->channel_id = channel_id;

        slot.is_ready.store(true, std::memory_order_release);
        m_write_index.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief Intercept and read available data slot directly without copying.
     */
    bool intercept_read_buffer(const uint8_t** out_payload_ptr, const StreamFrameHeader** out_header_ptr) {
        size_t current_r = m_read_index.load(std::memory_order_relaxed);
        size_t slot_idx = current_r % m_slot_count;
        BufferSlot& slot = m_slots[slot_idx];

        if (!slot.is_ready.load(std::memory_order_acquire)) {
            return false;
        }

        *out_header_ptr = slot.header;
        *out_payload_ptr = slot.payload;
        return true;
    }

    /**
     * @brief Release read slot after processing.
     */
    void release_read_buffer() {
        size_t current_r = m_read_index.load(std::memory_order_relaxed);
        size_t slot_idx = current_r % m_slot_count;
        BufferSlot& slot = m_slots[slot_idx];

        slot.is_ready.store(false, std::memory_order_release);
        m_read_index.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief Direct zero-copy transfer to output buffer.
     */
    bool direct_copy_to_float_buffer(float* dest, size_t max_floats, size_t* out_copied_floats) {
        const uint8_t* payload = nullptr;
        const StreamFrameHeader* header = nullptr;

        if (!intercept_read_buffer(&payload, &header)) {
            if (out_copied_floats) *out_copied_floats = 0;
            return false;
        }

        size_t num_floats = header->payload_bytes / sizeof(float);
        size_t to_copy = std::min(num_floats, max_floats);

        std::memcpy(dest, payload, to_copy * sizeof(float));
        if (out_copied_floats) *out_copied_floats = to_copy;

        release_read_buffer();
        return true;
    }

    [[nodiscard]] size_t slot_count() const noexcept { return m_slot_count; }
    [[nodiscard]] size_t max_payload_bytes() const noexcept { return m_max_payload_bytes; }
    [[nodiscard]] size_t dropped_frames() const noexcept { return m_dropped_frames.load(std::memory_order_relaxed); }

private:
    ArenaAllocator& m_arena;
    size_t m_slot_count{0};
    size_t m_max_payload_bytes{0};
    BufferSlot* m_slots{nullptr};

    std::atomic<size_t> m_write_index{0};
    std::atomic<size_t> m_read_index{0};
    std::atomic<size_t> m_dropped_frames{0};
};

} // namespace causal_engine
