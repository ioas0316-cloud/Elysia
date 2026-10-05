#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "elysia_scheduler.hpp"

void test_clarke_park_transformation() {
    std::cout << "[Test 1] Testing Clarke-Park Transformation & Inverse..." << std::endl;
    constexpr int N = 100;
    std::vector<float> abc(N * 3);
    std::vector<float> angles(N);
    std::vector<float> dq0(N * 3);

    for (int i = 0; i < N; ++i) {
        float theta = i * 0.05f;
        angles[i] = theta;
        abc[i * 3 + 0] = std::sin(theta);
        abc[i * 3 + 1] = std::sin(theta - 2.0f * M_PI / 3.0f);
        abc[i * 3 + 2] = std::sin(theta + 2.0f * M_PI / 3.0f);
    }

    launch_elysia_foc_kernel(abc.data(), angles.data(), dq0.data(), 1.0f, N);

    for (int i = 0; i < N; ++i) {
        float d = dq0[i * 3 + 0];
        float q = dq0[i * 3 + 1];
        float zero = dq0[i * 3 + 2];

        // Balanced 3-phase system zero component must be approximately 0
        assert(std::abs(zero) < 1e-4f);

        // Magnitude in DQ space must equal 1.5 * amplitude (balanced 3-phase property)
        float dq_mag = std::sqrt(d * d + q * q);
        assert(std::abs(dq_mag - 1.5f) < 1e-3f);
    }
    std::cout << "  >> PASSED!" << std::endl;
}

void test_flux_weakening_scaling() {
    std::cout << "[Test 2] Testing Flux Weakening Scaling..." << std::endl;
    ElysiaFOCScheduler scheduler(3072.0f, 0.15f, 0.005f);

    VRAMState normal_vram = VRAMTracker::get_simulated_state(2000.0f, 3072.0f);
    float gamma_normal = scheduler.compute_flux_weakening_factor(normal_vram);
    assert(gamma_normal == 1.0f);

    VRAMState high_vram = VRAMTracker::get_simulated_state(2900.0f, 3072.0f);
    float gamma_high = scheduler.compute_flux_weakening_factor(high_vram);
    assert(gamma_high < 1.0f && gamma_high >= 0.15f);

    std::cout << "  >> Gamma_D at 2000MB: " << gamma_normal << ", at 2900MB: " << gamma_high << " -> PASSED!" << std::endl;
}

void test_yoneda_embedding_and_stdp() {
    std::cout << "[Test 3] Testing Yoneda Embedding & STDP Phase Updates..." << std::endl;
    constexpr int K = 16;
    Multivector3D concept_A = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Multivector3D> basis(K, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<float> spectrum(K);

    launch_elysia_yoneda_embedding_kernel(&concept_A, basis.data(), spectrum.data(), K);
    for (int i = 0; i < K; ++i) {
        assert(spectrum[i] > 0.99f); // Identity rotor coherence should be ~1.0
    }

    std::vector<Rotor3D> rotors(K, {1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<float> t_pre(K, 10.0f);
    std::vector<float> t_post(K, 15.0f); // LTP delta_t = +5ms

    launch_elysia_stdp_rotor_kernel(rotors.data(), t_pre.data(), t_post.data(), 0.1f, 0.12f, 20.0f, 20.0f, K);
    for (int i = 0; i < K; ++i) {
        float norm = std::sqrt(rotors[i].s * rotors[i].s + rotors[i].b12 * rotors[i].b12 + rotors[i].b23 * rotors[i].b23 + rotors[i].b31 * rotors[i].b31);
        assert(std::abs(norm - 1.0f) < 1e-4f); // Must remain unit rotor
    }

    std::cout << "  >> PASSED!" << std::endl;
}

void test_3x3x3_rg_and_gaba_gating() {
    std::cout << "[Test 4] Testing 3x3x3 RG Coarse-Graining & GABA Gating..." << std::endl;
    constexpr int total_blocks = 4;
    std::vector<Multivector3D> micro_nodes(total_blocks * 27);
    std::vector<Multivector3D> macro_nodes(total_blocks);
    std::vector<float> coherence_factors(total_blocks);

    // Coherent alignment in block 0
    for (int i = 0; i < 27; ++i) {
        micro_nodes[i] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    launch_elysia_rg_3x3x3_kernel(micro_nodes.data(), macro_nodes.data(), coherence_factors.data(), total_blocks);
    assert(coherence_factors[0] > 0.95f); // High coherence factor for block 0

    launch_elysia_gaba_rg_gating_kernel(micro_nodes.data(), macro_nodes.data(), 0.35f, 10.0f, total_blocks);
    assert(macro_nodes[0].v1 > 0.0f); // Coherent block passes GABA gate

    std::cout << "  >> PASSED!" << std::endl;
}

void test_phase_crystallization() {
    std::cout << "[Test 5] Testing Phase Crystallization Transition..." << std::endl;
    constexpr int N = 8;
    std::vector<Rotor3D> dynamic_rotors(N, {0.9f, 0.1f, 0.0f, 0.0f});
    std::vector<Rotor3D> static_memory(N, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<unsigned char> is_crystallized(N, 0);

    launch_elysia_phase_crystallization_kernel(dynamic_rotors.data(), static_memory.data(), is_crystallized.data(), 0.7f, 0.8f, N);

    for (int i = 0; i < N; ++i) {
        assert(is_crystallized[i] == 1);
        float norm = std::sqrt(static_memory[i].s * static_memory[i].s + static_memory[i].b12 * static_memory[i].b12 + static_memory[i].b23 * static_memory[i].b23 + static_memory[i].b31 * static_memory[i].b31);
        assert(std::abs(norm - 1.0f) < 1e-4f);
    }
    std::cout << "  >> PASSED!" << std::endl;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Running Elysia FOC & Clifford Wave C++ Test Suite    \n";
    std::cout << "========================================================\n";

    test_clarke_park_transformation();
    test_flux_weakening_scaling();
    test_yoneda_embedding_and_stdp();
    test_3x3x3_rg_and_gaba_gating();
    test_phase_crystallization();

    std::cout << "========================================================\n";
    std::cout << "  ALL C++ FOC TESTS PASSED SUCCESSFULLY!               \n";
    std::cout << "========================================================\n";
    return 0;
}
