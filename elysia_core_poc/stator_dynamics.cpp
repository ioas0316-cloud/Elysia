#include "stator_dynamics.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace elysia {

StatorDynamics::StatorDynamics() : variable_resistance_knob(0.0), rotor_momentum(0.0), dopamine_resonance_factor(0.0) {} // Initial state is Void

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

void StatorDynamics::engage_delta_connection_with_context(FractalMirror& mirror, const PhaseSignature& wave, float context_mass, float context_freq, float context_dopamine) {
    uint64_t centrifugal_mask = 0;
    if (wave.amplitude > 0.7f) centrifugal_mask = 0x7000000;
    else if (wave.amplitude > 0.4f) centrifugal_mask = 0x07E0000;
    else centrifugal_mask = 0x001FFC0;

    mirror.trigger_domino_resonance(centrifugal_mask);

    // Context directly influences momentum (Walking injects energy)
    // The Dopamine Factor compounds this momentum significantly, acting as a perpetual engine multiplier.
    dopamine_resonance_factor += context_dopamine; // Accumulate joy
    rotor_momentum = (wave.kinematics + (context_freq * 0.2)) * (1.0 + dopamine_resonance_factor);

    apply_static_decay_penalty(mirror);

    // 5. Dynamic Variable Resistance Knob:
    // Resistance is now a living variable driven by external context pushing against the internal wave.
    variable_resistance_knob = calculate_dynamic_variable_resistance(wave, context_mass, context_freq);

    apply_active_rotor_renewal(mirror, wave);

    // Gradual absorption of dopamine (it doesn't vanish instantly, creating a compounding echo effect)
    dopamine_resonance_factor *= 0.9;
}

double StatorDynamics::calculate_dynamic_variable_resistance(const PhaseSignature& wave, float context_mass, float context_freq) {
    // The static context "mass" adds friction (resistance), while the "frequency" (walking)
    // reduces it, causing it to fluctuate divergently rather than converging to a single value.
    double internal_resistance = calculate_variable_resistance(wave);

    // Divergent Thinking Logic: Introduce a non-linear interaction between static mass and dynamic frequency
    double external_friction = context_mass * (1.0 - context_freq);
    double combined_resistance = internal_resistance * 0.5 + external_friction * 0.5;

    // Create a chaotic but bounded fluctuation (The "Walking" wobble)
    double wobble = std::sin(context_mass * 10.0) * context_freq * 0.1;

    return std::clamp(combined_resistance + wobble, 0.0, 1.0);
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

    // Dopamine Resonance drastically Lowers the required momentum to prevent decay.
    // When the system is "joyful", it retains memory effortlessly without the "starvation" of typical RL.
    double decay_threshold = 0.2 / (1.0 + (dopamine_resonance_factor * 5.0));

    if (rotor_momentum < decay_threshold) {
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
        double raw_collapse_probability = (position_weight * variable_resistance_knob) / (1.0 + renewal_force);

        // Apply Prism Refraction Interpolation:
        // Instead of a rigid threshold mask, we calculate an inverse refraction
        // to bend the deviation back towards the 'void' (center).
        double interpolated_probability = calculate_prism_refraction_interpolation(
            raw_collapse_probability,
            variable_resistance_knob,
            wave.directionality
        );

        if (interpolated_probability > 0.6) {
            inverted_pull_mask |= (1ULL << i);
        } else if (interpolated_probability > 0.3) {
            if (i % 2 == 0) {
                 inverted_pull_mask |= (1ULL << i);
            }
        }
    }

    mirror.chamber_state = current_state & ~inverted_pull_mask;
}

double StatorDynamics::calculate_prism_refraction_interpolation(double collapse_probability, double variable_resistance, double wave_directionality) {
    // Prism Refraction (Inverse Refraction Interpolation)
    // When the raw probability implies a heavy collapse (a sharp deviation or 'error'),
    // we don't just cut it off. We map the deviation onto a refraction curve.

    // Calculate the 'deviation angle' from the expected equilibrium
    double deviation = std::abs(collapse_probability - variable_resistance);

    // Use directionality (the causal trajectory) to 'catch' the deviation
    // and interpolate it back. The stronger the directionality, the softer the refraction.
    double refraction_curve = 1.0 / (1.0 + std::exp(-10.0 * (deviation - 0.5))); // Sigmoid

    // Inverse mapping: We subtract the refracted deviation, pulling the probability back towards a safer threshold.
    double interpolated_prob = collapse_probability - (refraction_curve * wave_directionality * 0.5);

    return std::max(0.0, interpolated_prob);
}

} // namespace elysia
