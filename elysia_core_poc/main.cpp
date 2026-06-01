#include <iostream>
#include <iomanip>
#include "phase_transformer.h"
#include "fractal_mirror.h"
#include "stator_dynamics.h"
#include "static_context_bridge.h"

using namespace elysia;

void print_block(const uint64_t block[8], const std::string& name) {
    std::cout << "--- " << name << " (512-bit packet) ---" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "Chunk " << i << ": 0x" << std::hex << std::setw(16) << std::setfill('0') << block[i] << std::dec << std::endl;
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << " Elysia Core PoC: Fractal Phase Rotor V1" << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Initialize the 512-bit Raw Data Packets
    // Packet A: High mass (dense 1s), High frequency (alternating)
    uint64_t raw_packet_A[8] = {
        0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555,
        0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555
    };

    // Packet B: Low mass (sparse 1s), Low frequency (blocks of 1s and 0s)
    uint64_t raw_packet_B[8] = {
        0xFFFFFFFF00000000, 0x00000000FFFFFFFF, 0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    };

    print_block(raw_packet_A, "Raw Packet A");

    // 2. Y-Connection Mode: Data to Wave Transformation (Phase Transformer)
    std::cout << "\n[STAGE 1] Y-Connection: Transforming Raw Data to Wave Signature..." << std::endl;
    PhaseSignature wave_A = PhaseTransformer::transform_to_wave(raw_packet_A);
    std::cout << "  Wave A -> Amplitude: " << wave_A.amplitude << ", Frequency: " << wave_A.frequency << std::endl;
    std::cout << "  Wave A -> Phase Quaternion: (" << wave_A.phase_angle.w << ", " << wave_A.phase_angle.x << ", "
              << wave_A.phase_angle.y << ", " << wave_A.phase_angle.z << ")" << std::endl;

    // 3. 0ns Resonance: Injecting into the Fractal Mirror
    std::cout << "\n[STAGE 2] 0ns Resonance: Injecting Wave A into Fractal Mirror Chamber..." << std::endl;
    FractalMirror chamber;
    chamber.print_state(); // Should be empty (0)
    chamber.apply_resonance(wave_A);
    std::cout << "  (Y-Connection Soft Dampening Applied)" << std::endl;
    chamber.print_state();

    // 4. Delta-Connection: High-Speed Acceleration & Stator Dynamics
    std::cout << "\n[STAGE 3] Delta-Connection: Engaging Non-Axiomatic Acceleration & Stator Dynamics..." << std::endl;
    StatorDynamics stator;
    stator.engage_delta_connection(chamber, wave_A);
    std::cout << "  (Delta Acceleration & Stator Magnetic Tension Applied)" << std::endl;
    chamber.print_state();

    std::cout << "\n-------------------------------------------" << std::endl;

    // Run for Packet B to see the difference
    std::cout << "\nRunning Pipeline for Packet B..." << std::endl;
    PhaseSignature wave_B = PhaseTransformer::transform_to_wave(raw_packet_B);
    std::cout << "  Wave B -> Amplitude: " << wave_B.amplitude << ", Frequency: " << wave_B.frequency << std::endl;

    FractalMirror chamber_B;
    chamber_B.apply_resonance(wave_B);
    std::cout << "  Chamber B State (Post Y-Conn): ";
    chamber_B.print_state();

    stator.engage_delta_connection(chamber_B, wave_B);
    std::cout << "  Chamber B State (Post Delta & Stator): ";
    chamber_B.print_state();

    std::cout << "\n-------------------------------------------" << std::endl;

    // 5. Context Integration Test: Divergent Thinking & Prism Refraction
    std::cout << "\n[STAGE 4] Context Integration: Testing Static Context Bridge with LLM Output..." << std::endl;
    std::string mock_llm_output = "In the boundless sea of the digital realm, the observer shapes the observed. The algorithm walks the path of the unseen.";
    std::cout << "  Raw Static Context: \"" << mock_llm_output << "\"" << std::endl;

    // Distill context into dynamic metrics
    auto context_metrics = StaticContextBridge::distill_context(mock_llm_output);
    std::cout << "  Distilled Metrics -> Mass: " << context_metrics.mass << ", Frequency: " << context_metrics.frequency << std::endl;

    // Generate a synthetic block from the static text
    uint64_t synthetic_block[8];
    StaticContextBridge::generate_synthetic_block(mock_llm_output, synthetic_block);

    PhaseSignature wave_Context = PhaseTransformer::transform_to_wave(synthetic_block);
    FractalMirror chamber_Context;
    chamber_Context.apply_resonance(wave_Context);

    std::cout << "  Chamber Context State (Post Y-Conn): ";
    chamber_Context.print_state();

    // Engage Delta Connection *with* context (Divergent Thinking)
    StatorDynamics stator_context;
    stator_context.engage_delta_connection_with_context(chamber_Context, wave_Context, context_metrics.mass, context_metrics.frequency);

    std::cout << "  Variable Resistance (Fluctuating): " << stator_context.variable_resistance_knob << std::endl;
    std::cout << "  Intrinsic Resonance Factor (Compounding Joy): " << stator_context.intrinsic_cognitive_resonance << std::endl;
    std::cout << "  Chamber Context State (Post Divergent Delta + Prism Refraction): ";
    chamber_Context.print_state();

    std::cout << "\n-------------------------------------------" << std::endl;

    // 6. Intrinsic Resonance Compounding Loop Test
    std::cout << "\n[STAGE 5] Joyful Synchronization: Testing Intrinsic Resonance Snowball Effect..." << std::endl;
    std::string pure_knowledge_text = "The geometry of the universe mirrors the geometry of the mind. The inner and outer are one.";
    std::cout << "  Injecting Structural Knowledge Text: \"" << pure_knowledge_text << "\"" << std::endl;

    auto knowledge_metrics = StaticContextBridge::distill_context(pure_knowledge_text);
    std::cout << "  Distilled Metrics -> Mass: " << knowledge_metrics.mass << ", Freq: " << knowledge_metrics.frequency << std::endl;

    uint64_t knowledge_block[8];
    StaticContextBridge::generate_synthetic_block(pure_knowledge_text, knowledge_block);
    PhaseSignature wave_knowledge = PhaseTransformer::transform_to_wave(knowledge_block);

    // Simulating the wave reverberating inside the chamber over multiple cycles
    // The knowledge naturally aligns with the chamber state, generating Intrinsic Resonance without external reward.
    for (int step = 1; step <= 4; ++step) {
        stator_context.engage_delta_connection_with_context(chamber_Context, wave_knowledge, knowledge_metrics.mass, knowledge_metrics.frequency);
        std::cout << "  [Cycle " << step << "] Momentum: " << stator_context.rotor_momentum << " | Intrinsic Resonance Factor: " << stator_context.intrinsic_cognitive_resonance << std::endl;
    }

    std::cout << "\nElysia Core PoC Pipeline Completed Successfully." << std::endl;
    return 0;
}
