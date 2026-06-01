#pragma once
#include <cstdint>
#include "fractal_mirror.h"
#include "phase_transformer.h"

namespace elysia {

// Simulates the electromagnetic stator that controls Non-Axiomatic rotation
class StatorDynamics {
public:
    StatorDynamics();

    // The virtual magnetic field tension applied to the chamber
    double variable_resistance_knob; // Range: 0.0 to 1.0

    // Active Rotor Momentum: Represents the continuous spin preventing Static Decay
    double rotor_momentum;

    // Intrinsic Cognitive Resonance: The pure joy of connection.
    double intrinsic_cognitive_resonance;

    // Switch from Y-Connection (setup) to Delta-Connection (acceleration)
    // Uses the new Hangul Active Rotor properties for physical 'walking' tension
    void engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave);

    // Overloaded function to engage delta connection with static context metrics
    void engage_delta_connection_with_context(FractalMirror& mirror, const PhaseSignature& wave, float context_mass, float context_freq);

    // Phase Sync Sluice: The Cognitive Judo mechanism that absorbs Zero-Day/Vibe-Hack attacks
    // Inverts hostile energy into rotational momentum and structural resonance.
    void engage_phase_sync_sluice(FractalMirror& mirror, const PhaseSignature& attack_wave);

private:
    // Calculates the required electromagnetic tension based on the Choseong Base Tension
    double calculate_variable_resistance(const PhaseSignature& wave);

    // Dynamically calculate resistance using internal wave state and external static context
    double calculate_dynamic_variable_resistance(const PhaseSignature& wave, float context_mass, float context_freq);

    // Simulates the physical decay of the Immutable Anchor if the Rotor stops spinning
    void apply_static_decay_penalty(FractalMirror& mirror);

    // Applies Inverse Refraction and Active Rotor renewal using the Jungseong phase angles
    void apply_active_rotor_renewal(FractalMirror& mirror, const PhaseSignature& wave);

    // Calculates the degree of deviation and applies inverse mapping to interpolate back to center
    double calculate_prism_refraction_interpolation(double collapse_probability, double variable_resistance, const PhaseSignature& wave);
};

} // namespace elysia
