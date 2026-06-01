#include "fractal_mirror.h"
#include <bitset>
#include <cmath>

namespace elysia {

FractalMirror::FractalMirror() : chamber_state(0) {}

void FractalMirror::apply_resonance(const PhaseSignature& wave) {
    // Map the wave's phase and frequency into a kernel mask.
    // The density (amplitude) and frequency dictates which mirrors are excited.

    // Create a deterministic bitmask from the quaternion and frequency.
    // This is a simplified mathematical mapping for the PoC.
    uint64_t base_excitation = 0;

    if (wave.frequency > 0.5f) {
        base_excitation |= 0xAAAAAAAA; // High frequency excites alternating mirrors
    } else {
        base_excitation |= 0x55555555;
    }

    // Use the Quaternion to shift/rotate the excitation mask
    // This represents the spatial addressing without pointers.
    int shift_amount = static_cast<int>(std::abs(wave.phase_angle.w) * 27.0f) % 27;

    uint64_t kernel_mask = (base_excitation << shift_amount) | (base_excitation >> (64 - shift_amount));

    // The Y-connection step: Soft application (AND/OR hybrid to simulate dampening/merging)
    // Here we just trigger the resonance directly as the wave enters.
    trigger_domino_resonance(kernel_mask);
}

void FractalMirror::trigger_domino_resonance(uint64_t kernel_mask) {
    // 0ns Resonance!
    // Instead of iterating through nodes or calling observer functions,
    // we apply a single SIMD-capable bitwise XOR to the entire spatial state.
    // The kernel_mask represents the interference pattern.

    // To ensure it affects the 3x3x3 (27 bits) space:
    uint64_t fractal_mask_27 = kernel_mask & 0x7FFFFFF; // 27 bits

    // The XOR operation flips the state of the mirror chamber instantly.
    chamber_state ^= fractal_mask_27;
}

void FractalMirror::print_state() const {
    std::cout << "Chamber State (27-bit): " << std::bitset<27>(chamber_state) << std::endl;
}

} // namespace elysia
