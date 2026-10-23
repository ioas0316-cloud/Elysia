#ifndef CAUSAL_ENGINE_CORE_COLLECTIVE_MANIFOLD_HPP
#define CAUSAL_ENGINE_CORE_COLLECTIVE_MANIFOLD_HPP

#include "superconducting_soa.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace causal_engine {

/**
 * @brief Symbiotic Protocell
 * Represents an individual informational protocell substrate (wrapper around SuperconductingSoAField)
 * with identity, self/other demarcation, causal deficit, and symbiotic coupling parameters.
 */
struct SymbioticProtocell {
    size_t id{0};
    SuperconductingSoAField field;
    float internal_energy{1.0f};
    float causal_deficit{0.0f};    // Measure of unfulfilled potential / friction / phase variance
    float self_identity_phase{0.0f}; // Microscopic identity phase signature
    float symbiotic_coupling{0.0f}; // Level of active alignment with other protocells

    SymbioticProtocell() = default;

    explicit SymbioticProtocell(size_t cell_id, size_t field_size)
        : id(cell_id),
          field(field_size),
          internal_energy(1.0f),
          causal_deficit(0.0f),
          self_identity_phase(static_cast<float>(cell_id) * 0.5f),
          symbiotic_coupling(0.0f) {}
};

/**
 * @brief Collective Manifold (Higher-order Emergent Manifold N -> N+1)
 * Encompasses an ensemble of SymbioticProtocells, managing pairwise Phase-Locking,
 * Symbiotic Alignment, Deficit Resolution, and topological inversion into a Collective Manifold.
 */
struct CollectiveManifold {
    std::vector<SymbioticProtocell> protocells;
    size_t system_dimension{1}; // Base dimension N

    // Emergent Collective (N -> N+1) Manifold Fields
    float collective_phase{0.0f};
    float collective_coherence{0.0f};
    float collective_macro_potential{0.0f};
    float topological_volume{1.0f};
    bool dimension_spawned{false}; // Set to true when N -> N+1 phase transition occurs

    CollectiveManifold() = default;

    explicit CollectiveManifold(size_t num_protocells, size_t cells_per_protocell) {
        protocells.reserve(num_protocells);
        for (size_t i = 0; i < num_protocells; ++i) {
            protocells.emplace_back(i, cells_per_protocell);
        }
    }

    void add_protocell(size_t field_size) {
        size_t new_id = protocells.size();
        protocells.emplace_back(new_id, field_size);
    }
};

/**
 * @brief Step Collective Manifold & Symbiotic Dynamics
 *
 * Computes:
 * 1. Individual protocell internal superconducting step.
 * 2. Causal Deficit evaluation (Self/Otherness contrast).
 * 3. Pairwise Symbiotic Alignment (Phase-locking across protocell boundaries).
 * 4. Emergent Collective Manifold synthesis & N -> N+1 topological expansion.
 */
inline void step_collective_manifold_dynamics(
    CollectiveManifold& manifold,
    float coupling_rate = 0.2f,
    float deficit_threshold = 0.0f,
    float dt = 0.1f
) {
    const size_t num_cells = manifold.protocells.size();
    if (num_cells == 0) return;

    // 1. Step individual protocells and compute local causal deficits
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < num_cells; ++i) {
        auto& cell = manifold.protocells[i];
        step_superconducting_transport(cell.field, 0.1f, 0.05f, 0.02f, 0.05f, dt);

        // Compute causal deficit from execution friction, phase variance, and uncoupled potential
        float total_friction = 0.0f;
        float phase_sum = 0.0f;
        size_t n = cell.field.num_cells;
        if (n > 0) {
            for (size_t k = 0; k < n; ++k) {
                // Symbiotic coupling actively reduces execution friction
                cell.field.execution_friction[k] *= (1.0f - 0.5f * cell.symbiotic_coupling);
                total_friction += cell.field.execution_friction[k];
                phase_sum += cell.field.signal_phase[k];
            }
            float avg_signal_phase = phase_sum / static_cast<float>(n);
            float phase_var = std::abs(cell.self_identity_phase - avg_signal_phase);
            cell.causal_deficit = (total_friction / static_cast<float>(n)) + phase_var + (1.0f - cell.symbiotic_coupling) * 0.1f;
        }
    }

    // 2. Inter-Protocell Symbiotic Alignment & Boundary Phase-Locking
    float total_inter_coherence = 0.0f;
    float sum_macro_potential = 0.0f;
    size_t pair_count = 0;

    for (size_t i = 0; i < num_cells; ++i) {
        for (size_t j = i + 1; j < num_cells; ++j) {
            auto& cell_a = manifold.protocells[i];
            auto& cell_b = manifold.protocells[j];

            // Otherness calculation: phase difference between self and other
            float phase_diff = std::abs(cell_a.self_identity_phase - cell_b.self_identity_phase);
            phase_diff = std::min(phase_diff, static_cast<float>(2.0f * M_PI) - phase_diff);

            // Symbiotic Complementarity: Deficit alignment occurs if there is mutual need & potential
            float mutual_deficit = cell_a.causal_deficit + cell_b.causal_deficit;

            if (mutual_deficit >= deficit_threshold) {
                // Phase-locking torque driving mutual alignment
                float alignment_force = std::sin(cell_b.self_identity_phase - cell_a.self_identity_phase);

                // Update self identity phase through symbiotic resonance
                float delta_a = coupling_rate * alignment_force * dt;
                cell_a.self_identity_phase += delta_a;
                cell_b.self_identity_phase -= delta_a;

                // Propagate phase alignment to signal phases in fields
                for (size_t k = 0; k < cell_a.field.num_cells; ++k) {
                    cell_a.field.signal_phase[k] += delta_a;
                }
                for (size_t k = 0; k < cell_b.field.num_cells; ++k) {
                    cell_b.field.signal_phase[k] -= delta_a;
                }

                // Symbiotic coupling level increases based on phase coherence
                float inter_coherence = std::cos(phase_diff * 0.5f);
                inter_coherence = std::max(0.0f, inter_coherence);

                cell_a.symbiotic_coupling = std::min(1.0f, cell_a.symbiotic_coupling + coupling_rate * inter_coherence * dt);
                cell_b.symbiotic_coupling = std::min(1.0f, cell_b.symbiotic_coupling + coupling_rate * inter_coherence * dt);

                total_inter_coherence += inter_coherence;
                pair_count++;
            }
        }
    }

    // 3. Emergent Collective Manifold Synthesis (N -> N+1 Topological Inversion)
    float avg_coherence = pair_count > 0 ? (total_inter_coherence / static_cast<float>(pair_count)) : 0.0f;
    manifold.collective_coherence = avg_coherence;

    // Aggregate macro potentials across protocells
    for (size_t i = 0; i < num_cells; ++i) {
        float cell_macro = 0.0f;
        for (size_t k = 0; k < manifold.protocells[i].field.num_cells; ++k) {
            cell_macro += manifold.protocells[i].field.macro_potential[k];
        }
        sum_macro_potential += cell_macro;
    }

    manifold.collective_macro_potential = sum_macro_potential;

    // Phase transition threshold check for N -> N+1 Dimension Spawning
    if (avg_coherence > 0.7f && num_cells >= 2) {
        if (!manifold.dimension_spawned) {
            manifold.dimension_spawned = true;
            manifold.system_dimension += 1; // Dimension Spawning N -> N+1
        }
        // Topological volume expansion driven by collective macro potential
        manifold.topological_volume = 1.0f + 0.5f * std::log(1.0f + std::abs(manifold.collective_macro_potential));

        // Topological Inversion: Collective manifold phase feedback back into individual protocells
        float avg_phase = 0.0f;
        for (size_t i = 0; i < num_cells; ++i) {
            avg_phase += manifold.protocells[i].self_identity_phase;
        }
        manifold.collective_phase = avg_phase / static_cast<float>(num_cells);

        // Guide protocells with collective phase
        for (size_t i = 0; i < num_cells; ++i) {
            float collective_pull = std::sin(manifold.collective_phase - manifold.protocells[i].self_identity_phase);
            manifold.protocells[i].self_identity_phase += 0.1f * collective_pull * dt;
        }
    } else {
        manifold.dimension_spawned = false;
        manifold.system_dimension = 1;
        manifold.topological_volume = 1.0f;
    }
}

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_COLLECTIVE_MANIFOLD_HPP
