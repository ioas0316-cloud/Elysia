#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>

#include "causal_engine/causal_bit_matrix.h"
#include "causal_engine/hyper_pixel.h"
#include "causal_engine/tensor_core_interop.h"

using namespace causal_engine;

void test_morton_code_encoding() {
    uint16_t x = 12;
    uint16_t y = 34;
    uint32_t code = MortonUtils::encode2D(x, y);
    assert(code > 0);

    uint64_t code64 = MortonUtils::encode2D_64(1024, 2048);
    assert(code64 > 0);
    std::cout << "[PASS] Morton Code Encoding Test\n";
}

void test_hyper_pixel_evaluation() {
    HyperPixelNode nodeA;
    nodeA.spatial_morton_code = 0xFF00FF00FF00FF00ULL;
    nodeA.causal_edge_mask    = 0xAAAAAAAAAAAAAAAAULL;
    nodeA.physics_state_flags = 0x5555555555555555ULL;
    nodeA.shader_voltage_data = 0xFFFFFFFF00000000ULL;

    HyperPixelNode nodeB;
    nodeB.spatial_morton_code = 0x00FF00FF00FF00FFULL;
    nodeB.causal_edge_mask    = 0xAAAAAAAAAAAAAAAAULL;
    nodeB.physics_state_flags = 0xFFFFFFFFFFFFFFFFULL;
    nodeB.shader_voltage_data = 0x00000000FFFFFFFFULL;

    nodeA.EvaluateCausalState(nodeB);

    assert(nodeA.spatial_morton_code == 0ULL);
    assert(nodeA.causal_edge_mask == 0xAAAAAAAAAAAAAAAAULL);
    assert(nodeA.physics_state_flags == 0x5555555555555555ULL);
    assert(nodeA.shader_voltage_data == 0ULL);

    std::cout << "[PASS] HyperPixelNode Evaluation Test\n";
}

void test_causal_bit_matrix_engine() {
    ZeroAbstractionCausalEngine<512, 512> engine;

    alignas(64) CausalBitMatrix<512, 512> sim_vram;
    engine.BindVRAMBuffer(&sim_vram);

    for (size_t y = 100; y < 150; ++y) {
        for (size_t x = 100; x < 150; ++x) {
            engine.FlipBit(x, y, 0); // Hitbox ON
            engine.FlipBit(x, y, 1); // Trajectory ON
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < 1000; ++frame) {
        engine.Tick();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed_ns = end - start;
    std::cout << "[PASS] CausalBitMatrix Engine 1000 Ticks Avg Latency: "
              << (elapsed_ns.count() / 1000.0) / 1000.0 << " us per frame\n";

    assert(sim_vram.frame_tick == 1000);
}

void test_early_culling() {
    uint64_t inactive_block = 0ULL;
    uint64_t active_block = 1ULL;

    assert(MortonQuadtreeCuller::ShouldCullBlock(inactive_block) == true);
    assert(MortonQuadtreeCuller::ShouldCullBlock(active_block) == false);

    std::cout << "[PASS] Morton Quadtree Early Culling Test\n";
}

int main() {
    std::cout << "=================================================\n";
    std::cout << " Running Hyper-Pixel & Zero-Abstraction Engine Tests\n";
    std::cout << "=================================================\n";

    test_morton_code_encoding();
    test_hyper_pixel_evaluation();
    test_causal_bit_matrix_engine();
    test_early_culling();

    std::cout << "=================================================\n";
    std::cout << " All Hyper-Pixel C++ Tests Completed Successfully\n";
    std::cout << "=================================================\n";
    return 0;
}
