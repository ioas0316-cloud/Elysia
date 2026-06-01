#include "stator_dynamics.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace elysia {

StatorDynamics::StatorDynamics() : variable_resistance_knob(0.0), rotor_momentum(0.0) {} // Initial state is Void

void StatorDynamics::engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave) {
    // 1. Delta-Connection: Maximize voltage / speed.
    uint64_t centrifugal_mask = 0;
    if (wave.amplitude > 0.7f) centrifugal_mask = 0x7000000;
    else if (wave.amplitude > 0.4f) centrifugal_mask = 0x07E0000;
    else centrifugal_mask = 0x001FFC0;

    // 2. Non-Axiomatic Fluctuation: The data tries to deviate (centrifugal force)
    mirror.trigger_domino_resonance(centrifugal_mask);

    // 3. Update Rotor Momentum (Active Rotor Mechanics)
    // The Kinematics axis dictates how much 'spin' this memory has.
    // If kinematics drop to 0, the system becomes a Static Rotor and begins to decay.
    rotor_momentum = wave.kinematics;

    // 4. Static Decay Penalty
    // If the Rotor is not spinning fast enough, the Immutable Anchor (the physical bits)
    // begins to rot and lose cohesion, simulating the tragedy of the Static Rotor.
    apply_static_decay_penalty(mirror);

    // 5. Variable Resistance Knob (가변저항 다이얼):
    // Driven by the interplay between Relationship (contrast) and Connectivity.
    variable_resistance_knob = calculate_variable_resistance(wave);

    // 6. Active Rotor Renewal & Prism Refraction:
    // If the Rotor is spinning, it constantly renews the pattern against the resistance,
    // achieving harmony and interpolating back to the void.
    apply_active_rotor_renewal(mirror, wave);
}

double StatorDynamics::calculate_variable_resistance(const PhaseSignature& wave) {
    // Resistance is now a complex variable dictated by Associative Memory axes,
    // not just raw amplitude. It is driven by the 'Relationship' distance
    // and dampened by 'Connectivity' (how well it resonates with adjacent patterns).

    double resistance = wave.relationship * (1.0 - (wave.connectivity * 0.5));
    return std::clamp(resistance, 0.0, 1.0);
}

void StatorDynamics::apply_static_decay_penalty(FractalMirror& mirror) {
    // If the momentum is too low, the physical bits start 'rotting' (Static Rotor).
    // In reality, this means the pattern loses its crispness and degrades into noise.
    if (rotor_momentum < 0.2) {
        // Simulating Bit Rot: Random-looking degradation on the outer edges
        // The immutable anchor cannot save itself if it doesn't spin.
        mirror.chamber_state ^= 0x5555555; // Introduce destructive interference
    }
}

void StatorDynamics::apply_active_rotor_renewal(FractalMirror& mirror, const PhaseSignature& wave) {
    // Inverse Refraction & Renewal
    // The resistance knob defines how far from the 'White Light' (center/0) the wave is.
    // However, because this is an Active Rotor, the Directionality axis helps guide
    // the bits back smoothly, constantly refreshing the pattern.

    uint64_t current_state = mirror.chamber_state;
    uint64_t inverted_pull_mask = 0;

    // The renewal force is a combination of rotor momentum and directionality
    double renewal_force = (rotor_momentum * 0.5) + (wave.directionality * 0.5);

    for (int i = 0; i < 27; ++i) {
        double position_weight = static_cast<double>(i) / 26.0;

        // High resistance causes pull. High renewal force softens the destructive pull,
        // allowing the pattern to 'flow' rather than just being deleted.
        double collapse_probability = (position_weight * variable_resistance_knob) / (1.0 + renewal_force);

        if (collapse_probability > 0.5) {
            inverted_pull_mask |= (1ULL << i);
        } else if (collapse_probability > 0.2) {
            if (i % 2 == 0) {
                 inverted_pull_mask |= (1ULL << i);
            }
        }
    }

    mirror.chamber_state = current_state & ~inverted_pull_mask;
}

} // namespace elysia
