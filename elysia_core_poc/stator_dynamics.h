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
    // Replaced fixed masks with a continuous damping coefficient (Variable Resistance)
    double variable_resistance_knob; // Range: 0.0 to 1.0

    // Active Rotor Momentum: Represents the continuous spin preventing Static Decay
    double rotor_momentum;

    // Intrinsic Cognitive Resonance: The pure joy of connection.
    // It is not injected externally, but naturally arises when incoming knowledge (wave)
    // perfectly aligns/resonates with the existing internal structure (chamber state).
    // This drives the compounding snowball effect without any artificial forced rewards.
    double intrinsic_cognitive_resonance;

    // Switch from Y-Connection (setup) to Delta-Connection (acceleration)
    // Now incorporates Active Rotor renewal logic against the Immutable Anchor
    void engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave);

    // Overloaded function to engage delta connection with static context metrics
    // This allows the variable resistance knob to be driven by external raw data.
    void engage_delta_connection_with_context(FractalMirror& mirror, const PhaseSignature& wave, float context_mass, float context_freq);

private:
    // Calculates the required electromagnetic tension (0.0 to 1.0) based on the associative axes
    double calculate_variable_resistance(const PhaseSignature& wave);

    // Dynamically calculate resistance using both internal wave state and external static context
    double calculate_dynamic_variable_resistance(const PhaseSignature& wave, float context_mass, float context_freq);

    // Simulates the physical decay of the Immutable Anchor if the Rotor stops spinning
    void apply_static_decay_penalty(FractalMirror& mirror);

    // Applies Inverse Refraction and Active Rotor renewal to smoothly interpolate back to equilibrium
    void apply_active_rotor_renewal(FractalMirror& mirror, const PhaseSignature& wave);

    // Calculates the degree of deviation and applies inverse mapping to interpolate back to center
    // This softens the destructive pull when the LLM data causes a sharp non-axiomatic deviation
    double calculate_prism_refraction_interpolation(double collapse_probability, double variable_resistance, double wave_directionality);
};

} // namespace elysia
