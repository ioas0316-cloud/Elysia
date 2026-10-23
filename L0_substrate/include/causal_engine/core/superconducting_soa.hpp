#ifndef CAUSAL_ENGINE_CORE_SUPERCONDUCTING_SOA_HPP
#define CAUSAL_ENGINE_CORE_SUPERCONDUCTING_SOA_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace causal_engine {

/**
 * @brief Superconducting SoA Field (1D Informational Protocell Substrate)
 *
 * Embeds phase-locking zero-scattering transport, demarcation boundary walls,
 * primordial telos energy gradients, irreversible macro potential hysteresis,
 * and self-attractor feedback loops.
 */
struct SuperconductingSoAField {
    size_t num_cells{0};

    // 1. Background Lattice Noise Field (Lattice Thermal Fluctuation)
    std::vector<float> lattice_phase;     // [0, 2pi]
    std::vector<float> lattice_freq;      // High frequency oscillation

    // 2. Transporting Signal Packet (Charge Carrier Wave)
    std::vector<float> signal_phase;      // Signal phase
    std::vector<float> signal_amplitude;  // Signal energy/amplitude

    // 3. Phase-Locking & Coherence States
    std::vector<float> phase_difference;  // |signal_phase - lattice_phase|
    std::vector<float> coherence_gate;    // Coherence ratio (0.0: high scattering, 1.0: zero-scattering)

    // 4. Protocell Demarcation & Telos & Hysteresis & Feedback
    std::vector<float> demarcation_wall;   // Boundary barrier (0.0: interior protocell, 1.0: boundary wall)
    std::vector<float> gradient_telos;     // Intrinsic directional energy gradient / bias
    std::vector<float> macro_potential;    // Irreversible hysteresis landscape
    std::vector<float> micro_velocity;     // Microscopic wave velocity / momentum
    std::vector<float> execution_friction; // Hardware / transport execution friction feedback

    SuperconductingSoAField() = default;

    explicit SuperconductingSoAField(size_t n)
        : num_cells(n),
          lattice_phase(n, 0.0f),
          lattice_freq(n, 0.1f),
          signal_phase(n, 0.0f),
          signal_amplitude(n, 0.0f),
          phase_difference(n, 0.0f),
          coherence_gate(n, 0.0f),
          demarcation_wall(n, 0.0f),
          gradient_telos(n, 0.0f),
          macro_potential(n, 0.0f),
          micro_velocity(n, 0.0f),
          execution_friction(n, 0.0f) {}

    void resize(size_t n) {
        num_cells = n;
        lattice_phase.assign(n, 0.0f);
        lattice_freq.assign(n, 0.1f);
        signal_phase.assign(n, 0.0f);
        signal_amplitude.assign(n, 0.0f);
        phase_difference.assign(n, 0.0f);
        coherence_gate.assign(n, 0.0f);
        demarcation_wall.assign(n, 0.0f);
        gradient_telos.assign(n, 0.0f);
        macro_potential.assign(n, 0.0f);
        micro_velocity.assign(n, 0.0f);
        execution_friction.assign(n, 0.0f);
    }
};

/**
 * @brief Zero-Scattering Transport & Protocell Dynamics Step
 */
inline void step_superconducting_transport(
    SuperconductingSoAField& field,
    float normal_damping_rate = 0.1f,
    float phase_lock_threshold = 0.05f,
    float hysteresis_rate = 0.02f,
    float feedback_strength = 0.05f,
    float dt = 0.1f
) {
    const size_t n = field.num_cells;
    if (n < 2) return;

    // 1. Parallel update for local phase, coherence, and local forces
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // 1.1 Background lattice thermal oscillation
        field.lattice_phase[i] = std::fmod(
            field.lattice_phase[i] + field.lattice_freq[i] * dt + static_cast<float>(2.0f * M_PI),
            static_cast<float>(2.0f * M_PI)
        );

        // 1.2 Telos gradient steering inside Demarcation boundary
        float is_interior = 1.0f - std::min(1.0f, std::max(0.0f, field.demarcation_wall[i]));
        if (is_interior > 0.01f) {
            // Telos pulls signal phase towards optimal alignment and drives velocity
            field.signal_phase[i] += field.gradient_telos[i] * is_interior * dt;
            field.micro_velocity[i] += field.gradient_telos[i] * 0.1f * is_interior * dt;
        }

        // Keep signal_phase within [0, 2pi)
        field.signal_phase[i] = std::fmod(
            field.signal_phase[i] + static_cast<float>(2.0f * M_PI),
            static_cast<float>(2.0f * M_PI)
        );

        // 1.3 Phase difference calculation |signal_phase - lattice_phase|
        float diff = std::abs(field.signal_phase[i] - field.lattice_phase[i]);
        field.phase_difference[i] = std::min(diff, static_cast<float>(2.0f * M_PI) - diff);

        // 1.4 Coherence Gate (Phase-Locking Transition)
        if (field.phase_difference[i] < phase_lock_threshold) {
            field.coherence_gate[i] = 1.0f;
            // Phase-Locking: lock signal phase perfectly with lattice phase
            field.signal_phase[i] = field.lattice_phase[i];
        } else {
            field.coherence_gate[i] *= 0.9f; // Gradual decay under decoherence
        }
    }

    // 2. Transport, Hysteresis, Self-Attractor Loop & Friction
    std::vector<float> next_amplitude = field.signal_amplitude;

    #pragma omp parallel for
    for (size_t i = 0; i < n - 1; ++i) {
        float active_coherence = field.coherence_gate[i];
        float wall_barrier = std::min(1.0f, std::max(0.0f, field.demarcation_wall[i]));

        // Zero-scattering transport permeability: Coherence gate bypasses normal damping
        float effective_damping = normal_damping_rate * (1.0f - active_coherence);

        // Demarcation wall blocks flow unless coherence is high (phase-locking opens channel through wall)
        float wall_permeability = (1.0f - wall_barrier) + wall_barrier * active_coherence;

        float effective_velocity = 0.5f + field.micro_velocity[i] * 0.1f;
        effective_velocity = std::min(1.0f, std::max(0.05f, effective_velocity));

        float flow_amount = field.signal_amplitude[i] * effective_velocity * wall_permeability;

        // Flow with zero friction under phase-locking
        float delivered_flow = flow_amount * (1.0f - effective_damping);
        float dissipated_energy = flow_amount * effective_damping;

        // Execution friction captures energy dissipation and phase mismatch
        field.execution_friction[i] = dissipated_energy + field.phase_difference[i] * 0.1f;

        // Energy transfer to neighbor
        #pragma omp atomic
        next_amplitude[i + 1] += delivered_flow;

        #pragma omp atomic
        next_amplitude[i] -= flow_amount;

        // 3. Irreversible Hysteresis & Macro Potential Engraving
        // Persistent coherent flow engraves macro potential landscape
        float flow_impact = delivered_flow * active_coherence;
        field.macro_potential[i] += hysteresis_rate * (flow_impact - 0.01f * field.macro_potential[i]);
        field.macro_potential[i] = std::min(10.0f, std::max(-10.0f, field.macro_potential[i]));
    }

    // 4. Apply Self-Attractor Feedback Loop
    // Spatial gradient of macro_potential pulls micro_velocity and signal_phase
    #pragma omp parallel for
    for (size_t i = 1; i < n - 1; ++i) {
        float macro_grad = (field.macro_potential[i + 1] - field.macro_potential[i - 1]) * 0.5f;

        // $-\nabla \text{macro\_potential}$ acts as self-attractor force
        field.micro_velocity[i] += -macro_grad * feedback_strength * dt;

        // Pull signal phase towards macro potential landscape attractor
        field.signal_phase[i] += -macro_grad * feedback_strength * 0.1f * dt;
    }

    // Numerical safety clamping for all elements
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(next_amplitude[i])) {
            next_amplitude[i] = 0.0f;
        } else {
            next_amplitude[i] = std::min(1e5f, std::max(0.0f, next_amplitude[i]));
        }
    }

    field.signal_amplitude = std::move(next_amplitude);
}

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_SUPERCONDUCTING_SOA_HPP
