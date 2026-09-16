#include "causal_engine/core/arena_allocator.hpp"
#include "causal_engine/core/zero_copy_interceptor.hpp"

#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>

using namespace causal_engine;

void test_arena_basic_allocation() {
    std::cout << "[Test 1] Testing ArenaAllocator basic bump-pointer allocation..." << std::endl;
    ArenaAllocator arena(1024 * 1024); // 1 MB slab

    assert(arena.capacity() == 1024 * 1024);
    assert(arena.used() == 0);

    int* p1 = arena.allocate_typed<int>(100);
    assert(p1 != nullptr);
    assert(arena.used() >= 100 * sizeof(int));
    assert(arena.allocation_count() == 1);

    for (int i = 0; i < 100; ++i) {
        p1[i] = i * 2;
    }
    assert(p1[99] == 198);

    arena.reset();
    assert(arena.used() == 0);
    assert(arena.allocation_count() == 0);

    std::cout << "  -> ArenaAllocator basic allocation passed successfully!" << std::endl;
}

void test_zero_copy_interceptor_streaming() {
    std::cout << "[Test 2] Testing ZeroCopyInterceptor pipeline streaming..." << std::endl;
    ArenaAllocator arena(4 * 1024 * 1024); // 4 MB slab
    ZeroCopyInterceptor interceptor(arena, 8, 256 * 1024); // 8 slots, 256 KB max payload

    uint8_t* write_payload = nullptr;
    StreamFrameHeader* write_header = nullptr;

    bool acquire_success = interceptor.acquire_write_buffer(&write_payload, &write_header);
    assert(acquire_success);
    assert(write_payload != nullptr);
    assert(write_header != nullptr);

    // Populate payload as floats
    float* float_ptr = reinterpret_cast<float*>(write_payload);
    for (size_t i = 0; i < 1000; ++i) {
        float_ptr[i] = static_cast<float>(i) * 1.5f;
    }

    interceptor.commit_write_buffer(1000 * sizeof(float), 1, 10);

    // Intercept and read
    const uint8_t* read_payload = nullptr;
    const StreamFrameHeader* read_header = nullptr;

    bool read_success = interceptor.intercept_read_buffer(&read_payload, &read_header);
    assert(read_success);
    assert(read_header->frame_index == 1);
    assert(read_header->payload_bytes == 1000 * sizeof(float));

    const float* read_floats = reinterpret_cast<const float*>(read_payload);
    assert(read_floats[999] == 999.0f * 1.5f);

    interceptor.release_read_buffer();

    std::cout << "  -> ZeroCopyInterceptor pipeline streaming passed successfully!" << std::endl;
}

void test_arena_benchmark() {
    std::cout << "[Test 3] Running sub-microsecond latency benchmark..." << std::endl;
    constexpr size_t ITERATIONS = 100000;
    ArenaAllocator arena(16 * 1024 * 1024);
    ZeroCopyInterceptor interceptor(arena, 32, 64 * 1024);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ITERATIONS; ++i) {
        uint8_t* payload = nullptr;
        StreamFrameHeader* header = nullptr;

        if (interceptor.acquire_write_buffer(&payload, &header)) {
            float* floats = reinterpret_cast<float*>(payload);
            floats[0] = static_cast<float>(i);
            interceptor.commit_write_buffer(sizeof(float), i, 1);
        }

        const uint8_t* r_payload = nullptr;
        const StreamFrameHeader* r_header = nullptr;
        if (interceptor.intercept_read_buffer(&r_payload, &r_header)) {
            interceptor.release_read_buffer();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / ITERATIONS;

    std::cout << "  -> Total time for " << ITERATIONS << " zero-copy stream cycles: "
              << total_us << " us (Avg: " << avg_us << " us / cycle)" << std::endl;
    assert(avg_us < 1.0); // Assert sub-microsecond latency per stream cycle
}

int main() {
    std::cout << "=== Running C++ Arena & Zero-Copy Interceptor Test Suite ===" << std::endl;
    test_arena_basic_allocation();
    test_zero_copy_interceptor_streaming();
    test_arena_benchmark();
    std::cout << "=== All C++ Arena & Zero-Copy Tests Passed Successfully! ===" << std::endl;
    return 0;
}
