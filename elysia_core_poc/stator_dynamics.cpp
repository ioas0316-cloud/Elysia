#include "stator_dynamics.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace elysia {

StatorDynamics::StatorDynamics() : variable_resistance_knob(0.0) {} // Initial resistance is zero (Pure Void)

void StatorDynamics::engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave) {
    // 1. Delta-Connection: Maximize voltage / speed.
    // In software, this means we amplify the phase angle to simulate high-speed rotation.
    // We create a "centrifugal" bitmask based on the wave's mass (amplitude).

    // High mass -> greater centrifugal force -> outward bit spread
    uint64_t centrifugal_mask = 0;
    if (wave.amplitude > 0.7f) {
        centrifugal_mask = 0x7000000; // Outer rings of the 27-bit space
    } else if (wave.amplitude > 0.4f) {
        centrifugal_mask = 0x07E0000; // Middle rings
    } else {
        centrifugal_mask = 0x001FFC0; // Inner rings
    }

    // 2. Non-Axiomatic Fluctuation: The data tries to deviate (centrifugal force)
    mirror.trigger_domino_resonance(centrifugal_mask);

    // 3. Variable Resistance Knob (가변저항 다이얼):
    // Instead of fixed thresholds, calculate a dynamic resistance factor (0.0 to 1.0)
    // based on the incoming wave's context (amplitude and frequency).
    variable_resistance_knob = calculate_variable_resistance(wave);

    // 4. Prism Refraction (프리즘 보간 수문):
    // Smoothly interpolate the distorted state back towards equilibrium (the void / 0)
    // using the calculated resistance knob.
    // This avoids the violent collisions of hard bitwise masking.
    apply_prism_refraction(mirror, variable_resistance_knob);
}

double StatorDynamics::calculate_variable_resistance(const PhaseSignature& wave) {
    // The resistance (knob position) smoothly scales with the 'noise' or distortion.
    // We combine amplitude (mass) and frequency (volatility) into a continuous function.
    // This makes the system a *variable* that adapts to the wave, rather than a rigid wall.

    // Base resistance is derived from wave intensity
    double intensity = std::sqrt(wave.amplitude * wave.amplitude + wave.frequency * wave.frequency);

    // Normalize and map to a 0.0 - 1.0 range (sigmoid-like smooth clamping)
    double resistance = intensity / (1.0 + intensity);

    // Ensure bounds
    return std::clamp(resistance, 0.0, 1.0);
}

void StatorDynamics::apply_prism_refraction(FractalMirror& mirror, double resistance_knob) {
    // Inverse Refraction: The resistance knob defines how far from the 'White Light' (center/0) the wave is.
    // Instead of cutting off the data with hard thresholds, we smoothly pull it back based on the resistance.
    // We perform continuous interpolation by scaling the impact of the resistance on the chamber bits.

    uint64_t current_state = mirror.chamber_state;

    // We probabilistically flip bits back to 0 (The Void) based on the continuous resistance knob.
    // A resistance of 1.0 means strong suppression of non-axiomatic deviation.
    // A resistance of 0.0 means perfect harmony, letting the wave flow naturally.
    // This removes the "if-else" fixed barriers completely.

    uint64_t inverted_pull_mask = 0;

    // Iterate through the 27 bits of the chamber state
    for (int i = 0; i < 27; ++i) {
        // Outer rings (higher bit index) have inherently higher kinetic energy,
        // meaning they need stronger dampening if resistance is high.
        double position_weight = static_cast<double>(i) / 26.0;

        // The probability to collapse this bit to 0 is proportional to its distance from center
        // multiplied by the global resistance knob.
        double collapse_probability = position_weight * resistance_knob;

        // We simulate a pseudo-continuous fade. In a pure analog system this would be a voltage drop.
        // Here, we create a dynamic mask that softens the wave rather than hard-clipping it.
        // Since we don't have RNG here, we use deterministic modulo based on position to simulate a gradient mask
        if (collapse_probability > 0.5) {
            // Strong pull: The bit is designated to be masked out
            inverted_pull_mask |= (1ULL << i);
        } else if (collapse_probability > 0.1) {
            // Soft pull: The bit is 'sometimes' masked (we simulate this by skipping every other bit)
            if (i % 2 == 0) {
                 inverted_pull_mask |= (1ULL << i);
            }
        }
    }

    // Apply the smooth gradient mask. ~inverted_pull_mask means bits marked with 1 are forced to 0 (pulled to Void).
    mirror.chamber_state = current_state & ~inverted_pull_mask;
}

} // namespace elysia
