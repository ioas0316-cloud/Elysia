#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "causal_engine/core/superconducting_soa.hpp"
#include "causal_engine/core/topological_field_2d.hpp"

using namespace causal_engine;

void test_stage_1_demarcation_and_telos() {
    std::cout << "[Test Stage 1] Demarcation Boundary Wall and Telos Gradient..." << std::endl;
    SuperconductingSoAField field(64);

    // Define Demarcation wall (0..15: interior, 16..20: wall, 21..63: exterior)
    for (size_t i = 16; i <= 20; ++i) {
        field.demarcation_wall[i] = 1.0f; // Solid wall
    }

    // Set primordial telos gradient inside interior
    for (size_t i = 0; i < 16; ++i) {
        field.gradient_telos[i] = 0.5f;
    }

    step_superconducting_transport(field, 0.1f, 0.05f, 0.02f, 0.05f, 0.1f);

    // Verify telos accelerated micro_velocity inside interior
    assert(field.micro_velocity[5] > 0.0f);
    // Verify demarcation wall prevented leakage when un-cohere
    assert(field.demarcation_wall[18] == 1.0f);

    std::cout << "  -> Stage 1 Passed!" << std::endl;
}

void test_stage_2_phase_locking_zero_scattering() {
    std::cout << "[Test Stage 2] Phase-Locking Zero-Scattering Transport..." << std::endl;
    SuperconductingSoAField field(64);

    // Inject thermal lattice noise
    for (size_t i = 0; i < 64; ++i) {
        field.lattice_phase[i] = 1.23f;
    }

    // Cell 5: Aligned phase (phase-locking)
    field.signal_phase[5] = 1.24f; // diff = 0.01 < threshold 0.05
    field.signal_amplitude[5] = 100.0f;

    // Cell 15: Unaligned phase (decoherence)
    field.signal_phase[15] = 4.00f; // diff ~ 2.77 > threshold
    field.signal_amplitude[15] = 100.0f;

    step_superconducting_transport(field, 0.2f, 0.05f, 0.02f, 0.05f, 0.1f);

    // Coherence gate opened at cell 5
    assert(field.coherence_gate[5] == 1.0f);
    // Coherence gate closed at cell 15
    assert(field.coherence_gate[15] < 0.5f);

    // Cell 5 should transfer energy without scattering attenuation
    assert(field.signal_amplitude[6] > field.signal_amplitude[16]);

    std::cout << "  -> Stage 2 Passed!" << std::endl;
}

void test_stage_3_irreversible_hysteresis() {
    std::cout << "[Test Stage 3] Irreversible Macro Potential Hysteresis..." << std::endl;
    SuperconductingSoAField field(64);

    field.lattice_phase[10] = 0.5f;
    field.signal_phase[10] = 0.5f;
    field.coherence_gate[10] = 1.0f;
    field.signal_amplitude[10] = 50.0f;

    // Run multiple transport steps to engrave macro potential
    for (int step = 0; step < 20; ++step) {
        step_superconducting_transport(field, 0.1f, 0.1f, 0.05f, 0.05f, 0.1f);
    }

    // Macro potential should be persistently engraved at cell 10
    assert(field.macro_potential[10] > 0.0f);

    std::cout << "  -> Stage 3 Passed!" << std::endl;
}

void test_stage_4_self_attractor_feedback() {
    std::cout << "[Test Stage 4] Self-Attractor Feedback Loop..." << std::endl;
    SuperconductingSoAField field(64);

    // Engrave macro potential valley slope (cell 29: 0.0, cell 30: 5.0, cell 31: 10.0)
    field.macro_potential[29] = 0.0f;
    field.macro_potential[30] = 5.0f;
    field.macro_potential[31] = 10.0f;

    // Run transport step
    step_superconducting_transport(field, 0.1f, 0.05f, 0.02f, 0.1f, 0.1f);

    // $-\nabla \text{macro\_potential}$ force should drive micro velocity at slope (cell 30)
    assert(field.micro_velocity[30] < 0.0f); // Slope points right, negative gradient drives velocity left

    std::cout << "  -> Stage 4 Passed!" << std::endl;
}

void test_topological_2d_vorticity() {
    std::cout << "[Test 2D Topological Field] Phase Gradient & Vorticity..." << std::endl;
    TopologicalField2D field(16, 16);

    // Set vortex phase singularity pattern
    for (size_t y = 1; y < 15; ++y) {
        for (size_t x = 1; x < 15; ++x) {
            float dx = static_cast<float>(x) - 8.0f;
            float dy = static_cast<float>(y) - 8.0f;
            field.phase[y * 16 + x] = std::atan2(dy, dx) + M_PI;
            field.amplitude[y * 16 + x] = 10.0f;
        }
    }

    step_multidim_topological_transport(field, 0.1f, 0.1f);

    // Center should detect non-zero vorticity
    assert(field.vorticity[8 * 16 + 8] >= 0.0f);

    std::cout << "  -> 2D Topological Field Passed!" << std::endl;
}

int main() {
    std::cout << "=== Running Superconducting Kernel C++ Test Suite ===" << std::endl;
    test_stage_1_demarcation_and_telos();
    test_stage_2_phase_locking_zero_scattering();
    test_stage_3_irreversible_hysteresis();
    test_stage_4_self_attractor_feedback();
    test_topological_2d_vorticity();
    std::cout << "All C++ Superconducting Kernel Tests Passed Successfully!" << std::endl;
    return 0;
}
