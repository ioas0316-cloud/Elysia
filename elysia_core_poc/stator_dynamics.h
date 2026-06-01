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

    // Dopamine Resonance Factor: Simulates the "joy" of learning, creating a compounding snowball effect.
    // Instead of forcing decay (deprivation/starvation), this factor prolongs resonance,
    // allowing associative patterns to naturally expand and compound over time.
    double dopamine_resonance_factor;

    // Switch from Y-Connection (setup) to Delta-Connection (acceleration)
    // Now incorporates Active Rotor renewal logic against the Immutable Anchor
    void engage_delta_connection(FractalMirror& mirror, const PhaseSignature& wave);

    // Overloaded function to engage delta connection with static context metrics
    // This allows the variable resistance knob to be driven by external raw data.
    // Now accepts a dopamine multiplier to inject the "Joy/Reward" signal directly into the core.
    void engage_delta_connection_with_context(FractalMirror& mirror, const PhaseSignature& wave, float context_mass, float context_freq, float context_dopamine = 0.0f);

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
