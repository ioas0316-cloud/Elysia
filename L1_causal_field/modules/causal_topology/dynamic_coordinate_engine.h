#ifndef DYNAMIC_COORDINATE_ENGINE_H
#define DYNAMIC_COORDINATE_ENGINE_H

#include "causal_trace_pool.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace causal_topology {

struct EngineConfig {
    float condensation_threshold = 0.8f; // Invariance score above which variable x condenses into an Axis (1,2)
    float relativization_friction = 0.5f;  // Friction/prediction error threshold above which Axis softens back into Variable x
    float min_resistor_x = 0.001f;        // Minimum impedance (direct connection / tight causality)
    float max_resistor_x = 100.0f;        // Maximum impedance (decoupled / high uncertainty)
    float dt = 0.01f;                     // Time step
    float learning_rate = 0.1f;
};

class DynamicCoordinateEngine {
private:
    EngineConfig config_;

public:
    DynamicCoordinateEngine() = default;
    explicit DynamicCoordinateEngine(const EngineConfig& config) : config_(config) {}

    const EngineConfig& get_config() const { return config_; }
    void set_config(const EngineConfig& config) { config_ = config; }

    // Phase 1: Forward Phase (Compute state update through causal circuit)
    // Signal passes through variable resistance dial resistor_x
    // I_effective = delta_input / resistor_x
    void forward_step(CausalStatePool& pool, const std::vector<float>& inputs) {
        size_t n = std::min(pool.count, inputs.size());
        for (size_t i = 0; i < n; ++i) {
            float eff_res = pool.is_axis[i] ? config_.min_resistor_x : std::max(pool.resistor_x[i], config_.min_resistor_x);
            float effective_force = inputs[i] / eff_res;

            // State delta
            pool.deltas[i] = effective_force * config_.dt;
            pool.values[i] += pool.deltas[i];
        }
    }

    // Phase 2: Trace Capture Phase
    // Log the current state transition into Causal Trace Pool
    size_t capture_trace(CausalStatePool& pool, size_t node_idx, uint32_t op_id, uint32_t ctx_mask, int32_t parent_idx) {
        if (node_idx >= pool.count) {
            throw std::out_of_range("node_idx out of range in capture_trace");
        }
        return pool.push_back(
            pool.values[node_idx],
            pool.deltas[node_idx],
            op_id,
            ctx_mask,
            pool.invariance_scores[node_idx],
            parent_idx,
            pool.resistor_x[node_idx],
            pool.is_axis[node_idx]
        );
    }

    // Phase 3: Reflective Metaprocess Phase (Relativization & Condensation)
    // Updates invariance scores and shifts dynamic coordinates ($1,2 \leftrightarrow x$)
    // When do-intervention / prediction error is low across contexts -> Invariance increases -> Condenses to Axis
    // When friction / contradiction is detected on an Axis -> Softens back to Variable x
    void reflect_and_mutate(CausalStatePool& pool, const std::vector<float>& prediction_errors) {
        size_t n = std::min(pool.count, prediction_errors.size());
        for (size_t i = 0; i < n; ++i) {
            float err = std::abs(prediction_errors[i]);

            if (pool.is_axis[i]) {
                // (1) Check for Relativization: Axis (1,2) -> Variable x
                // If friction/error on a rigid axis exceeds threshold, relativize into dial x
                if (err > config_.relativization_friction) {
                    pool.is_axis[i] = 0;
                    // Increase impedance x to reflect renewed uncertainty
                    pool.resistor_x[i] = std::min(config_.max_resistor_x, pool.resistor_x[i] + config_.learning_rate * err + 0.5f);
                    pool.invariance_scores[i] = std::max(0.0f, pool.invariance_scores[i] - 0.2f);
                } else {
                    // Axis holds firm
                    pool.invariance_scores[i] = std::min(1.0f, pool.invariance_scores[i] + 0.05f);
                }
            } else {
                // (2) Variable x dynamic adaptation & Condensation Check: Variable x -> Axis
                if (err < 0.1f) {
                    // Invariance score increases as error remains small
                    pool.invariance_scores[i] = std::min(1.0f, pool.invariance_scores[i] + 0.05f + config_.learning_rate * (0.1f - err));
                    // Tighten resistance (lower x) toward direct connection
                    pool.resistor_x[i] = std::max(config_.min_resistor_x, pool.resistor_x[i] * 0.9f);
                } else {
                    // Friction increases impedance x
                    pool.resistor_x[i] = std::min(config_.max_resistor_x, pool.resistor_x[i] + config_.learning_rate * err);
                    pool.invariance_scores[i] = std::max(0.0f, pool.invariance_scores[i] - config_.learning_rate * err);
                }

                // Condense to Axis if invariance score exceeds threshold
                if (pool.invariance_scores[i] >= config_.condensation_threshold) {
                    pool.is_axis[i] = 1;
                    pool.resistor_x[i] = config_.min_resistor_x; // Direct rigid connection
                }
            }
        }
    }
};

} // namespace causal_topology

#endif // DYNAMIC_COORDINATE_ENGINE_H
