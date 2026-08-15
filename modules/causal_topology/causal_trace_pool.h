#ifndef CAUSAL_TRACE_POOL_H
#define CAUSAL_TRACE_POOL_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace causal_topology {

// SoA (Structure of Arrays) Memory Pool for Causal Trace & Invariance History
// Stores state, trace, operator ID, context, invariance score, parent links, and variable resistance (impedance x).
struct CausalStatePool {
    size_t count = 0;

    // 1. State and Dynamics
    std::vector<float> values;            // Current state value X_t
    std::vector<float> deltas;            // State delta dX / dt or change amount

    // 2. Process Trace & Operator Metadata
    std::vector<uint32_t> operator_ids;   // Applied operator / mechanism ID
    std::vector<uint32_t> context_masks;  // Context/boundary/constraint condition mask at operation time

    // 3. Causal Invariance & Back-tracing
    std::vector<float> invariance_scores; // Invariance score under repeated do-interventions
    std::vector<int32_t> parent_trace_idx;// Parent trace index that induced this change (-1 if root)

    // 4. Dynamic Coordinate System (Axis vs Variable Resistance x)
    std::vector<float> resistor_x;        // Variable impedance / resistance x (uncertainty / degree of freedom dial)
    std::vector<uint8_t> is_axis;          // 1 if fixed as rigid Coordinate Axis / Lens (1, 2), 0 if dynamic variable dial x

    CausalStatePool() = default;

    explicit CausalStatePool(size_t capacity) {
        resize(capacity);
    }

    void resize(size_t new_size) {
        count = new_size;
        values.resize(new_size, 0.0f);
        deltas.resize(new_size, 0.0f);
        operator_ids.resize(new_size, 0);
        context_masks.resize(new_size, 0);
        invariance_scores.resize(new_size, 0.0f);
        parent_trace_idx.resize(new_size, -1);
        resistor_x.resize(new_size, 1.0f); // Default impedance 1.0f
        is_axis.resize(new_size, 0);        // Default released as variable x
    }

    void reserve(size_t capacity) {
        values.reserve(capacity);
        deltas.reserve(capacity);
        operator_ids.reserve(capacity);
        context_masks.reserve(capacity);
        invariance_scores.reserve(capacity);
        parent_trace_idx.reserve(capacity);
        resistor_x.reserve(capacity);
        is_axis.reserve(capacity);
    }

    size_t push_back(float val, float delta, uint32_t op_id, uint32_t ctx_mask,
                      float inv_score, int32_t parent_idx, float res_x = 1.0f, uint8_t axis_flag = 0) {
        values.push_back(val);
        deltas.push_back(delta);
        operator_ids.push_back(op_id);
        context_masks.push_back(ctx_mask);
        invariance_scores.push_back(inv_score);
        parent_trace_idx.push_back(parent_idx);
        resistor_x.push_back(res_x);
        is_axis.push_back(axis_flag);
        count = values.size();
        return count - 1;
    }

    void clear() {
        count = 0;
        values.clear();
        deltas.clear();
        operator_ids.clear();
        context_masks.clear();
        invariance_scores.clear();
        parent_trace_idx.clear();
        resistor_x.clear();
        is_axis.clear();
    }

    // Back-trace the causal chain starting from trace_idx to root
    std::vector<int32_t> trace_back(int32_t start_idx) const {
        std::vector<int32_t> chain;
        int32_t curr = start_idx;
        while (curr >= 0 && static_cast<size_t>(curr) < count) {
            chain.push_back(curr);
            curr = parent_trace_idx[curr];
        }
        return chain;
    }
};

} // namespace causal_topology

#endif // CAUSAL_TRACE_POOL_H
