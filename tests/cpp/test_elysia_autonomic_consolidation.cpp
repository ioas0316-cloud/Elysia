#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>
#include "elysia_autonomic_controller.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

float calculate_phase_entropy(const std::vector<float2>& rotors, int num_bins = 64) {
    std::vector<int> histogram(num_bins, 0);
    int valid_rotors = 0;

    for (const auto& r : rotors) {
        float mag = std::sqrt(r.x * r.x + r.y * r.y);
        if (mag < 1e-5f) continue; // Pruned rotors (zero)

        float phase = std::atan2(r.y, r.x); // (-pi, pi]
        float norm_phase = (phase + M_PI) / (2.0f * M_PI); // [0.0, 1.0)
        int bin = std::min(num_bins - 1, static_cast<int>(norm_phase * num_bins));

        histogram[bin]++;
        valid_rotors++;
    }

    if (valid_rotors == 0) return 0.0f;

    float entropy = 0.0f;
    for (int count : histogram) {
        if (count > 0) {
            float p = static_cast<float>(count) / valid_rotors;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

float calculate_mean_coherence(const std::vector<float>& coherence_map) {
    double sum = 0.0;
    for (float c : coherence_map) {
        sum += c;
    }
    return static_cast<float>(sum / coherence_map.size());
}

void test_autonomic_state_machine() {
    std::cout << "[Test 1] Testing Autonomic State Machine Transitions..." << std::endl;
    AutonomicStateController controller;

    assert(controller.get_mode() == AutonomicMode::Parasympathetic);

    // Mismatch spike triggers Sympathetic transition
    EngineMetrics spike = {0.80f, 0.20f, 0.10f};
    controller.update(spike);
    assert(controller.get_mode() == AutonomicMode::Sympathetic);
    assert(controller.get_ach_level() == 1.0f);

    // Mismatch drops & coherence recovers triggers Parasympathetic transition
    EngineMetrics coherent = {0.10f, 0.85f, 0.10f};
    controller.update(coherent);
    assert(controller.get_mode() == AutonomicMode::Parasympathetic);
    assert(controller.get_ach_level() == 0.02f);

    std::cout << "  >> PASSED!" << std::endl;
}

void test_vram_adaptive_controller() {
    std::cout << "[Test 2] Testing VRAM Adaptive Controller..." << std::endl;
    VramAdaptiveController controller;

    // Simulated 50% free VRAM (safe state) -> gentle threshold ~0.05
    controller.set_simulated_vram_free_ratio(0.50f);
    assert(std::abs(controller.get_dynamic_threshold() - 0.05f) < 1e-3f);

    // Simulated 10% free VRAM (critical state) -> aggressive threshold ~0.45
    controller.set_simulated_vram_free_ratio(0.10f);
    assert(std::abs(controller.get_dynamic_threshold() - 0.45f) < 1e-3f);

    std::cout << "  >> PASSED!" << std::endl;
}

void test_parasympathetic_consolidation() {
    std::cout << "[Test 3] Testing Parasympathetic Consolidation Benchmark..." << std::endl;
    constexpr int NUM_ROTORS = 10000;
    std::vector<float2> rotors(NUM_ROTORS);
    std::vector<float> coherence_map(NUM_ROTORS, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> phase_dist(-M_PI, M_PI);
    std::uniform_real_distribution<float> mag_dist(0.01f, 1.5f);

    for (int i = 0; i < NUM_ROTORS; ++i) {
        float p = phase_dist(rng);
        float m = mag_dist(rng);
        rotors[i] = make_float2(m * std::cos(p), m * std::sin(p));
    }

    float pre_entropy = calculate_phase_entropy(rotors);
    float pre_coherence = calculate_mean_coherence(coherence_map);

    launch_parasympathetic_consolidation_kernel(rotors.data(), coherence_map.data(), 0.01f, 0.15f, NUM_ROTORS);

    float post_entropy = calculate_phase_entropy(rotors);
    float post_coherence = calculate_mean_coherence(coherence_map);

    assert(post_entropy < pre_entropy);      // Entropy must decrease
    assert(post_coherence > pre_coherence);  // Mean coherence must increase

    std::cout << "  >> Pre Entropy: " << pre_entropy << " bits -> Post Entropy: " << post_entropy << " bits (Reduced)" << std::endl;
    std::cout << "  >> Pre Coherence: " << pre_coherence << " -> Post Coherence: " << post_coherence << " (Boosted)" << std::endl;
    std::cout << "  >> PASSED!" << std::endl;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Running Elysia Autonomic Consolidation Benchmark Suite \n";
    std::cout << "========================================================\n";

    test_autonomic_state_machine();
    test_vram_adaptive_controller();
    test_parasympathetic_consolidation();

    std::cout << "========================================================\n";
    std::cout << "  ALL AUTONOMIC CONSOLIDATION TESTS PASSED!             \n";
    std::cout << "========================================================\n";
    return 0;
}
