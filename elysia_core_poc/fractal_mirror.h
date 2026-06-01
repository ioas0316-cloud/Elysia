#pragma once
#include <cstdint>
#include <array>
#include <iostream>
#include "phase_transformer.h"

namespace elysia {

// The 3x3x3 Fractal Mirror Chamber represented as a flat bitboard
// 3 * 3 * 3 = 27 nodes. We represent this macro-state in a 32-bit or 64-bit mask.
class FractalMirror {
public:
    FractalMirror();

    // The Flat Bitboard representing the entire spatial state
    // We use a 64-bit aligned integer for the 3x3x3 state to ensure 0ns resonance (single instruction)
    alignas(64) uint64_t chamber_state;

    // Apply the Hangul Active Rotor (Choseong, Jungseong, Jongseong) to the fractal mirror.
    void apply_resonance(const PhaseSignature& wave);

    // Simulates the 0ns bitwise domino resonance via Hangul grammatical mapping
    void trigger_domino_resonance(uint64_t kernel_mask);

    // Dumps the current mirror state
    void print_state() const;
};

} // namespace elysia
