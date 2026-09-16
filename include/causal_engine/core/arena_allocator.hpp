#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <atomic>
#include <memory>
#include <algorithm>
#include <string>

namespace causal_engine {

/**
 * @brief High-performance lock-free / bump-pointer pre-allocated Arena Allocator.
 *
 * Guarantees zero dynamic allocations during runtime loops by serving memory
 * from a contiguous pre-allocated slab.
 */
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity_bytes, size_t default_alignment = 64)
        : m_capacity(capacity_bytes), m_alignment(default_alignment), m_offset(0), m_allocation_count(0) {
        if (capacity_bytes == 0) {
            throw std::invalid_argument("Arena capacity must be greater than 0");
        }

        // Allocate cache-line aligned memory block
#if defined(_MSC_VER) || defined(__MINGW32__)
        m_buffer = static_cast<uint8_t*>(_aligned_malloc(m_capacity, m_alignment));
#else
        if (posix_memalign(reinterpret_cast<void**>(&m_buffer), std::max<size_t>(m_alignment, sizeof(void*)), m_capacity) != 0) {
            m_buffer = nullptr;
        }
#endif
        if (!m_buffer) {
            throw std::bad_alloc();
        }
    }

    ~ArenaAllocator() {
        if (m_buffer) {
#if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(m_buffer);
#else
            std::free(m_buffer);
#endif
            m_buffer = nullptr;
        }
    }

    // Disable copy
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Enable move
    ArenaAllocator(ArenaAllocator&& other) noexcept
        : m_buffer(other.m_buffer), m_capacity(other.m_capacity),
          m_alignment(other.m_alignment), m_offset(other.m_offset.load()),
          m_allocation_count(other.m_allocation_count.load()) {
        other.m_buffer = nullptr;
        other.m_capacity = 0;
        other.m_offset = 0;
        other.m_allocation_count = 0;
    }

    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept {
        if (this != &other) {
            if (m_buffer) {
#if defined(_MSC_VER) || defined(__MINGW32__)
                _aligned_free(m_buffer);
#else
                std::free(m_buffer);
#endif
            }
            m_buffer = other.m_buffer;
            m_capacity = other.m_capacity;
            m_alignment = other.m_alignment;
            m_offset = other.m_offset.load();
            m_allocation_count = other.m_allocation_count.load();

            other.m_buffer = nullptr;
            other.m_capacity = 0;
            other.m_offset = 0;
            other.m_allocation_count = 0;
        }
        return *this;
    }

    /**
     * @brief Allocate contiguous memory from the arena slab.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment constraint (must be power of 2).
     * @return Pointer to allocated memory inside the arena.
     */
    void* allocate(size_t bytes, size_t alignment = 0) {
        if (bytes == 0) return nullptr;
        if (alignment == 0) alignment = m_alignment;

        size_t current_offset = m_offset.load(std::memory_order_relaxed);
        while (true) {
            uintptr_t current_ptr = reinterpret_cast<uintptr_t>(m_buffer + current_offset);
            size_t align_mask = alignment - 1;
            uintptr_t aligned_ptr = (current_ptr + align_mask) & ~align_mask;
            size_t new_offset = (aligned_ptr - reinterpret_cast<uintptr_t>(m_buffer)) + bytes;

            if (new_offset > m_capacity) {
                throw std::runtime_error("ArenaAllocator out of memory: requested " +
                                         std::to_string(bytes) + " bytes, remaining " +
                                         std::to_string(m_capacity - current_offset));
            }

            if (m_offset.compare_exchange_weak(current_offset, new_offset,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
                m_allocation_count.fetch_add(1, std::memory_order_relaxed);
                return reinterpret_cast<void*>(aligned_ptr);
            }
        }
    }

    /**
     * @brief Typed allocation helper.
     */
    template <typename T>
    T* allocate_typed(size_t count = 1) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    /**
     * @brief Fast reset: rewinds bump pointer to zero without releasing underlying slab memory.
     */
    void reset() noexcept {
        m_offset.store(0, std::memory_order_release);
        m_allocation_count.store(0, std::memory_order_release);
    }

    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }
    [[nodiscard]] size_t used() const noexcept { return m_offset.load(std::memory_order_relaxed); }
    [[nodiscard]] size_t remaining() const noexcept { return m_capacity - used(); }
    [[nodiscard]] size_t allocation_count() const noexcept { return m_allocation_count.load(std::memory_order_relaxed); }
    [[nodiscard]] uint8_t* raw_buffer() noexcept { return m_buffer; }
    [[nodiscard]] const uint8_t* raw_buffer() const noexcept { return m_buffer; }

private:
    uint8_t* m_buffer{nullptr};
    size_t m_capacity{0};
    size_t m_alignment{64};
    std::atomic<size_t> m_offset{0};
    std::atomic<size_t> m_allocation_count{0};
};

} // namespace causal_engine
