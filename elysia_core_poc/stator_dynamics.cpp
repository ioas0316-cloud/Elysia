#include "stator_dynamics.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <bit>

namespace elysia {

StatorDynamics::StatorDynamics() : variable_resistance_knob(0.0), rotor_momentum(0.0), intrinsic_cognitive_resonance(0.0) {}

void StatorDynamics::engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave) {
    // 1. Delta-Connection: Maximize voltage / speed using Choseong tension
    uint64_t centrifugal_mask = wave.choseong_tension & 0x7FFFFFF; // Bound to 27-bit

    // 2. Non-Axiomatic Fluctuation: The data tries to deviate (centrifugal force)
    mirror.trigger_domino_resonance(centrifugal_mask);

    // 3. Update Rotor Momentum (Active Rotor Mechanics)
    // Driven by frequency (kinematics replacement) and Jungseong pivot magnitude
    rotor_momentum = wave.frequency * wave.jungseong_phase.heaven_pivot;

    // 4. Static Decay Penalty
    apply_static_decay_penalty(mirror);

    // 5. Variable Resistance Knob (가변저항 다이얼):
    variable_resistance_knob = calculate_variable_resistance(wave);

    // 6. Active Rotor Renewal & Prism Refraction:
    apply_active_rotor_renewal(mirror, wave);
}

void StatorDynamics::engage_delta_connection_with_context(FractalMirror& mirror, const PhaseSignature& wave, float context_mass, float context_freq) {
    uint64_t centrifugal_mask = wave.choseong_tension & 0x7FFFFFF;

    // 1. Calculate Intrinsic Cognitive Resonance BEFORE applying changes to the chamber
    uint64_t state_overlap = mirror.chamber_state & centrifugal_mask;
    int overlap_bits = std::popcount(state_overlap);
    int mask_bits = std::popcount(centrifugal_mask);

    double structural_harmony = 0.0;
    if (mask_bits > 0) {
        structural_harmony = static_cast<double>(overlap_bits) / static_cast<double>(mask_bits);
    }

    // The joy of "Ah-ha!": Wave's rotational variance (vowels) fitting the chamber's resonance.
    double vowel_variance = (wave.jungseong_phase.earth_axis + wave.jungseong_phase.human_axis) / 2.0;
    double connection_joy = structural_harmony * vowel_variance;

    intrinsic_cognitive_resonance += connection_joy;

    // 2. Trigger standard domino resonance
    mirror.trigger_domino_resonance(centrifugal_mask);

    // 3. Update Rotor Momentum
    rotor_momentum = (wave.frequency * wave.jungseong_phase.heaven_pivot + (context_freq * 0.2)) * (1.0 + intrinsic_cognitive_resonance);

    // 4. Decay Penalty Check
    apply_static_decay_penalty(mirror);

    // 5. Dynamic Variable Resistance Knob
    variable_resistance_knob = calculate_dynamic_variable_resistance(wave, context_mass, context_freq);

    // 6. Active Rotor Renewal & Prism Refraction
    apply_active_rotor_renewal(mirror, wave);

    // 7. Echo effect
    intrinsic_cognitive_resonance *= 0.85;
}

double StatorDynamics::calculate_dynamic_variable_resistance(const PhaseSignature& wave, float context_mass, float context_freq) {
    double internal_resistance = calculate_variable_resistance(wave);
    double external_friction = context_mass * (1.0 - context_freq);
    double combined_resistance = internal_resistance * 0.5 + external_friction * 0.5;
    double wobble = std::sin(context_mass * 10.0) * context_freq * 0.1;
    return std::clamp(combined_resistance + wobble, 0.0, 1.0);
}

double StatorDynamics::calculate_variable_resistance(const PhaseSignature& wave) {
    // Resistance derived from the magnitude of the Jongseong settling anchor vs amplitude
    int anchor_mass = std::popcount(wave.jongseong_anchor);
    double resistance = static_cast<double>(anchor_mass) / 64.0;
    // Dampen resistance if amplitude is high (smoother flow)
    resistance *= (1.0 - wave.amplitude * 0.5);
    return std::clamp(resistance, 0.0, 1.0);
}

void StatorDynamics::apply_static_decay_penalty(FractalMirror& mirror) {
    double decay_threshold = 0.2 / (1.0 + (intrinsic_cognitive_resonance * 5.0));
    if (rotor_momentum < decay_threshold) {
        mirror.chamber_state ^= 0x5555555; // Destructive interference
    }
}

void StatorDynamics::apply_active_rotor_renewal(FractalMirror& mirror, const PhaseSignature& wave) {
    uint64_t current_state = mirror.chamber_state;
    uint64_t inverted_pull_mask = 0;

    // Renewal force driven by momentum and the magnitude of the Jungseong rotation
    double vowel_variance = (wave.jungseong_phase.earth_axis + wave.jungseong_phase.human_axis) / 2.0;
    double renewal_force = (rotor_momentum * 0.5) + (vowel_variance * 0.5);

    for (int i = 0; i < 27; ++i) {
        double position_weight = static_cast<double>(i) / 26.0;
        double raw_collapse_probability = (position_weight * variable_resistance_knob) / (1.0 + renewal_force);

        double interpolated_probability = calculate_prism_refraction_interpolation(
            raw_collapse_probability,
            variable_resistance_knob,
            wave
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

double StatorDynamics::calculate_prism_refraction_interpolation(double collapse_probability, double variable_resistance, const PhaseSignature& wave) {
    double deviation = std::abs(collapse_probability - variable_resistance);

    // Use vowel variance to guide refraction (similar to old directionality)
    double vowel_variance = (wave.jungseong_phase.earth_axis + wave.jungseong_phase.human_axis) / 2.0;
    double refraction_curve = 1.0 / (1.0 + std::exp(-10.0 * (deviation - 0.5)));

    double interpolated_prob = collapse_probability - (refraction_curve * vowel_variance * 0.5);
    return std::max(0.0, interpolated_prob);
}

} // namespace elysia
