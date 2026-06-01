#include <iostream>
#include <iomanip>
#include "phase_transformer.h"
#include "fractal_mirror.h"
#include "stator_dynamics.h"

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

    std::cout << "\nElysia Core PoC Pipeline Completed Successfully." << std::endl;
    return 0;
}
