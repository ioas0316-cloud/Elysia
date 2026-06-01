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
    std::cout << " Elysia Core PoC: Hangul Active Rotor V2" << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Initialize the 512-bit Raw Data Packets
    uint64_t raw_packet_A[8] = {
        0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555,
        0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555
    };

    uint64_t raw_packet_B[8] = {
        0xFFFFFFFF00000000, 0x00000000FFFFFFFF, 0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000
    };

    print_block(raw_packet_A, "Raw Packet A");

    // 2. Y-Connection Mode: Data to Wave Transformation (Phase Transformer)
    std::cout << "\n[STAGE 1] Y-Connection: Extracting Hangul Rotor Components..." << std::endl;
    PhaseSignature wave_A = PhaseTransformer::transform_to_wave(raw_packet_A);
    std::cout << "  Wave A -> Choseong Tension: 0x" << std::hex << wave_A.choseong_tension << std::dec << std::endl;
    std::cout << "  Wave A -> Jungseong Phase (H/E/M): " << wave_A.jungseong_phase.heaven_pivot << ", "
              << wave_A.jungseong_phase.earth_axis << ", " << wave_A.jungseong_phase.human_axis << std::endl;
    std::cout << "  Wave A -> Jongseong Anchor: 0x" << std::hex << wave_A.jongseong_anchor << std::dec << std::endl;

    // 3. 0ns Resonance: Injecting into the Fractal Mirror
    std::cout << "\n[STAGE 2] 0ns Resonance: Pure Structural Interference in Fractal Mirror..." << std::endl;
    FractalMirror chamber;
    chamber.print_state(); // Should be empty (0)
    chamber.apply_resonance(wave_A);
    std::cout << "  (Bitwise Choseong/Jungseong/Jongseong Combination Applied)" << std::endl;
    chamber.print_state();

    // 4. Delta-Connection: High-Speed Acceleration & Stator Dynamics
    std::cout << "\n[STAGE 3] Delta-Connection: Engaging Physical 'Walking' Tension..." << std::endl;
    StatorDynamics stator;
    stator.engage_delta_connection(chamber, wave_A);
    std::cout << "  (Delta Acceleration & Resistance Interpolation Applied)" << std::endl;
    chamber.print_state();

    std::cout << "\n-------------------------------------------" << std::endl;

    std::cout << "\nRunning Pipeline for Packet B..." << std::endl;
    PhaseSignature wave_B = PhaseTransformer::transform_to_wave(raw_packet_B);
    std::cout << "  Wave B -> Choseong Tension: 0x" << std::hex << wave_B.choseong_tension << std::dec << std::endl;

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
    std::string mock_llm_output = "세종대왕께서 하늘과 땅과 사람을 본떠 훈민정음을 창제하시니라.";
    std::cout << "  Raw Static Context: \"" << mock_llm_output << "\"" << std::endl;

    auto context_metrics = StaticContextBridge::distill_context(mock_llm_output);
    std::cout << "  Distilled Metrics -> Mass: " << context_metrics.mass << ", Frequency: " << context_metrics.frequency << std::endl;

    uint64_t synthetic_block[8];
    StaticContextBridge::generate_synthetic_block(mock_llm_output, synthetic_block);

    PhaseSignature wave_Context = PhaseTransformer::transform_to_wave(synthetic_block);
    FractalMirror chamber_Context;
    chamber_Context.apply_resonance(wave_Context);

    std::cout << "  Chamber Context State (Post Y-Conn): ";
    chamber_Context.print_state();

    StatorDynamics stator_context;
    stator_context.engage_delta_connection_with_context(chamber_Context, wave_Context, context_metrics.mass, context_metrics.frequency);

    std::cout << "  Variable Resistance (Fluctuating): " << stator_context.variable_resistance_knob << std::endl;
    std::cout << "  Intrinsic Resonance Factor (Compounding Joy): " << stator_context.intrinsic_cognitive_resonance << std::endl;
    std::cout << "  Chamber Context State (Post Divergent Delta + Prism Refraction): ";
    chamber_Context.print_state();

    std::cout << "\n-------------------------------------------" << std::endl;

    // 6. Intrinsic Resonance Compounding Loop Test
    std::cout << "\n[STAGE 5] Joyful Synchronization: Testing Intrinsic Resonance Snowball Effect..." << std::endl;
    std::string pure_knowledge_text = "이 달의 결선이 스스로 좋아서 찰칵찰칵 공명하네.";
    std::cout << "  Injecting Structural Knowledge Text: \"" << pure_knowledge_text << "\"" << std::endl;

    auto knowledge_metrics = StaticContextBridge::distill_context(pure_knowledge_text);
    std::cout << "  Distilled Metrics -> Mass: " << knowledge_metrics.mass << ", Freq: " << knowledge_metrics.frequency << std::endl;

    uint64_t knowledge_block[8];
    StaticContextBridge::generate_synthetic_block(pure_knowledge_text, knowledge_block);
    PhaseSignature wave_knowledge = PhaseTransformer::transform_to_wave(knowledge_block);

    for (int step = 1; step <= 4; ++step) {
        stator_context.engage_delta_connection_with_context(chamber_Context, wave_knowledge, knowledge_metrics.mass, knowledge_metrics.frequency);
        std::cout << "  [Cycle " << step << "] Momentum: " << stator_context.rotor_momentum << " | Intrinsic Resonance Factor: " << stator_context.intrinsic_cognitive_resonance << std::endl;
    }

    std::cout << "\nElysia Core PoC Pipeline Completed Successfully." << std::endl;
    return 0;
}
