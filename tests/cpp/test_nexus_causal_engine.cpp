#include <iostream>
#include <cassert>
#include <vector>
#include "nexus_betti.h"
#include "nexus_in_engine_ai.h"
#include "nexus_bitwise_audio.h"

void TestBettiComputation() {
    std::cout << "[CPP Test] Testing 2D Betti Number Calculation...\n";
    // 4x4 Grid with 1 solid 2x2 square (betti_0 = 1, betti_1 = 0)
    std::vector<uint8_t> grid_solid = {
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 0
    };
    CausalNexus::BettiNumbers b_solid = CausalNexus::CalculateBetti2D(grid_solid, 4, 4);
    assert(b_solid.betti_0 == 1);
    assert(b_solid.betti_1 == 0);

    // 5x5 Grid forming a ring/hole in the center (betti_0 = 1, betti_1 = 1)
    std::vector<uint8_t> grid_ring = {
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 0, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
    };
    CausalNexus::BettiNumbers b_ring = CausalNexus::CalculateBetti2D(grid_ring, 5, 5);
    assert(b_ring.betti_0 == 1);
    assert(b_ring.betti_1 == 1);

    std::cout << "  ✓ Betti 2D computation test passed! (Solid b0="
              << b_solid.betti_0 << ", b1=" << b_solid.betti_1
              << " | Ring b0=" << b_ring.betti_0 << ", b1=" << b_ring.betti_1 << ")\n";
}

void TestInEngineAIBridge() {
    std::cout << "[CPP Test] Testing In-Engine Direct3D12 AI Bridge & Ping-Pong Swapchain...\n";
    CausalNexus::InEngineCausalAIBridge bridge(16, 16);

    std::vector<uint8_t> causal_mask(16 * 16, 0);
    // Fill 4x4 area in center
    for (int r = 4; r < 8; ++r) {
        for (int c = 4; c < 8; ++c) {
            causal_mask[r * 16 + c] = 1;
        }
    }

    CausalNexus::BettiNumbers expected_betti{1, 0};
    CausalNexus::FenceStatus status = bridge.ExecuteCausalInferenceAndBindDX12(causal_mask, 16, 16, expected_betti);
    assert(status == CausalNexus::FenceStatus::SIGNALED);
    assert(bridge.GetFenceValue() == 1);

    // Test Topological Rollback when betti number fails
    CausalNexus::BettiNumbers invalid_expected_betti{99, 99};
    CausalNexus::FenceStatus rollback_status = bridge.ExecuteCausalInferenceAndBindDX12(causal_mask, 16, 16, invalid_expected_betti);
    assert(rollback_status == CausalNexus::FenceStatus::TOPOLOGICAL_ROLLBACK);

    std::cout << "  ✓ In-Engine AI Bridge test passed!\n";
}

void TestBitwiseAudioVisual() {
    std::cout << "[CPP Test] Testing Bitwise BNN & Audio-Visual Voltage Surround...\n";
    std::vector<uint64_t> inputs = {0xFFFFFFFFFFFFFFFFULL};
    std::vector<uint64_t> weights = {0xFFFFFFFFFFFFFFFFULL};

    std::vector<uint8_t> act = CausalNexus::BitwiseCausalSimulator::ExecuteBitwiseXnorPopcnt(inputs, weights);
    assert(act.size() == 64);
    for (uint8_t a : act) {
        assert(a == 1);
    }

    std::vector<uint8_t> traj(16 * 16, 0);
    std::vector<uint8_t> hit(16 * 16, 0);
    traj[4 * 16 + 2] = 1; // Left side
    hit[8 * 16 + 12] = 1; // Right side

    auto frame = CausalNexus::BitwiseCausalSimulator::SplitAudioVisualVoltage(traj, hit, 16, 16);
    assert(frame.visual_voltage_rgb.size() == 16 * 16 * 3);
    assert(frame.audio_dsp_registers[0] >= 0.0f); // Left channel
    assert(frame.audio_dsp_registers[1] >= 0.0f); // Right channel

    std::cout << "  ✓ Bitwise BNN & Audio-Visual Voltage Surround test passed!\n";
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "   RUNNING CAUSAL NEXUS C++ SUITE\n";
    std::cout << "=================================================\n";
    TestBettiComputation();
    TestInEngineAIBridge();
    TestBitwiseAudioVisual();
    std::cout << "ALL C++ TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
