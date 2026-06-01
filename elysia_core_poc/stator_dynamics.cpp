#include "stator_dynamics.h"

namespace elysia {

StatorDynamics::StatorDynamics() : stator_field_mask(0xFFFFFFF) {} // Full tension by default (27 bits)

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

    // 3. Stator Tension: The virtual electromagnetic stator applies tension to pull it back
    // to the Axiomatic center (void/equilibrium).
    stator_field_mask = calculate_stator_tension(wave);

    // Apply the stator field using a bitwise AND/XOR hybrid to simulate magnetic pulling
    // (Correcting the deviation back towards 0 or a stable pattern).
    // This is the software equivalent of the magnetic field grabbing the wandering wave.

    // AND with inverse stator mask pulls it back towards 0 (the void/center).
    mirror.chamber_state &= ~stator_field_mask;
}

uint64_t StatorDynamics::calculate_stator_tension(const PhaseSignature& wave) {
    // Stator tension is proportional to frequency (resonance) and amplitude (mass)
    // If the wave is too wild, the stator generates a stronger counter-field.

    uint64_t tension = 0;
    if (wave.frequency > 0.8f || wave.amplitude > 0.8f) {
        // High instability -> Strong Stator field (masking out the outer edges)
        tension = 0x7FE0000; // Strong pull on outer/middle nodes
    } else {
        // Normal operation -> Mild Stator field
        tension = 0x7000000; // Gentle pull on only the outermost nodes
    }

    return tension;
}

} // namespace elysia
