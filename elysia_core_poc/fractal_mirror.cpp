#include "fractal_mirror.h"
#include <bitset>
#include <cmath>

namespace elysia {

FractalMirror::FractalMirror() : chamber_state(0) {}

void FractalMirror::apply_resonance(const PhaseSignature& wave) {
    // 1. Choseong (X-axis divergence): The initial base tension
    // This provides the raw excitation pattern (e.g. ㄱ, ㄴ, ㅁ)
    uint64_t base_excitation = wave.choseong_tension;

    // 2. Jungseong (Y-axis coupling): Vowels act as phase angles for rotation
    // Heaven (w) controls the magnitude of shift
    // Earth (x) and Human (z) influence the rotational variance
    float rotation_factor = wave.jungseong_phase.heaven_pivot *
                            (1.0f + wave.jungseong_phase.earth_axis + wave.jungseong_phase.human_axis);

    // Shift amount bounded within the 27-bit (3x3x3) mirror chamber
    int shift_amount = static_cast<int>(rotation_factor * 27.0f) % 27;

    // Apply rotation (Bitwise shift simulating spatial torque).
    // Uses C++20 std::rotl to avoid undefined behavior if shift_amount == 0
    uint64_t rotated_excitation = std::rotl(base_excitation, shift_amount);

    // 3. Jongseong (Z-axis settling): The anchoring mask
    // Fuses the rotated wave with the grounding tension
    uint64_t kernel_mask = rotated_excitation ^ wave.jongseong_anchor;

    // Apply the 0ns resonance
    trigger_domino_resonance(kernel_mask);
}

void FractalMirror::trigger_domino_resonance(uint64_t kernel_mask) {
    // 0ns Resonance!
    // No 'if' statements, no mapping dictionaries.
    // The Choseong + Jungseong + Jongseong phase state perfectly collapses into one of the
    // emergent states in the 27-bit space using purely structural bitwise interference.

    uint64_t fractal_mask_27 = kernel_mask & 0x7FFFFFF; // Clamp to 27 bits (3x3x3)

    // The XOR operation flips the state of the mirror chamber instantly,
    // simulating pure cognitive wave interference.
    chamber_state ^= fractal_mask_27;
}

void FractalMirror::print_state() const {
    std::cout << "Chamber State (27-bit): " << std::bitset<27>(chamber_state) << std::endl;
}

} // namespace elysia
