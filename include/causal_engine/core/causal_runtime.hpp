#ifndef CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP
#define CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <atomic>
#include <memory>
#include <cmath>
#include <algorithm>
#include <functional>

namespace causal_engine {
namespace core {

// =========================================================================
// 1. Causal Signal & Event Lock-Free Ring Buffer
// =========================================================================
struct CausalSignal {
    uint64_t signal_id = 0;
    uint32_t source_node_id = 0;
    uint32_t target_node_id = 0;
    float magnitude = 1.0f;
    float radius = 10.0f;
    float position[3] = {0.0f, 0.0f, 0.0f};
    uint64_t timestamp_ns = 0;
};

template <size_t Capacity = 1024>
class LockFreeEventRingBuffer {
private:
    CausalSignal buffer_[Capacity];
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

public:
    LockFreeEventRingBuffer() = default;

    bool enqueue(const CausalSignal& signal) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        buffer_[current_tail] = signal;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool dequeue(CausalSignal& signal) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }

        signal = buffer_[current_head];
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_relaxed);
        return (t >= h) ? (t - h) : (Capacity - h + t);
    }
};

// =========================================================================
// 2. Structural Causal Model (SCM) & Node Structural Equations
// =========================================================================
struct SCMNode {
    uint32_t id = 0;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float factual_state = 0.0f;         // Actual factual state (S)
    float counterfactual_state = 0.0f;  // Counterfactual state S_{do(X)}
    float exogenous_noise = 0.0f;       // U_i noise term
    bool is_intervened = false;         // Indicates do(X = x) applied
    float intervention_value = 0.0f;
    bool manifested = false;            // Causal collapse state
    uint64_t last_evaluated_tick = 0;

    std::vector<uint32_t> parents;      // Incoming causal edges (causes)
    std::vector<uint32_t> children;     // Outgoing causal edges (effects)

    // Structural equation function Y = f_Y(Parents(Y), U_Y)
    std::function<float(const std::vector<float>&, float)> structural_eq;
};

class StructuralCausalModel {
private:
    std::unordered_map<uint32_t, SCMNode> nodes_;

public:
    void add_node(
        uint32_t id,
        float x, float y, float z,
        std::function<float(const std::vector<float>&, float)> eq = nullptr)
    {
        SCMNode node;
        node.id = id;
        node.position[0] = x;
        node.position[1] = y;
        node.position[2] = z;
        node.structural_eq = eq ? eq : [](const std::vector<float>& parents, float noise) {
            float sum = noise;
            for (float p : parents) sum += p * 0.8f;
            return sum;
        };
        nodes_[id] = node;
    }

    bool add_causal_edge(uint32_t cause_id, uint32_t effect_id) {
        if (nodes_.find(cause_id) == nodes_.end() || nodes_.find(effect_id) == nodes_.end()) {
            return false;
        }
        if (has_causal_path(effect_id, cause_id)) {
            // Cycle prevention in SCM DAG
            return false;
        }

        nodes_[cause_id].children.push_back(effect_id);
        nodes_[effect_id].parents.push_back(cause_id);
        return true;
    }

    bool has_causal_path(uint32_t start, uint32_t target) const {
        if (start == target) return true;
        std::unordered_set<uint32_t> visited;
        std::queue<uint32_t> q;
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            uint32_t curr = q.front();
            q.pop();

            auto it = nodes_.find(curr);
            if (it == nodes_.end()) continue;

            for (uint32_t child : it->second.children) {
                if (child == target) return true;
                if (visited.find(child) == visited.end()) {
                    visited.insert(child);
                    q.push(child);
                }
            }
        }
        return false;
    }

    SCMNode* get_node(uint32_t id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    const std::unordered_map<uint32_t, SCMNode>& get_all_nodes() const {
        return nodes_;
    }

    std::vector<uint32_t> get_topological_order() const {
        std::unordered_map<uint32_t, int> in_degree;
        for (const auto& pair : nodes_) {
            in_degree[pair.first] = static_cast<int>(pair.second.parents.size());
        }

        std::queue<uint32_t> q;
        for (const auto& pair : in_degree) {
            if (pair.second == 0) {
                q.push(pair.first);
            }
        }

        std::vector<uint32_t> topo;
        while (!q.empty()) {
            uint32_t curr = q.front();
            q.pop();
            topo.push_back(curr);

            auto it = nodes_.find(curr);
            if (it != nodes_.end()) {
                for (uint32_t child : it->second.children) {
                    in_degree[child]--;
                    if (in_degree[child] == 0) {
                        q.push(child);
                    }
                }
            }
        }

        for (const auto& pair : nodes_) {
            if (std::find(topo.begin(), topo.end(), pair.first) == topo.end()) {
                topo.push_back(pair.first);
            }
        }
        return topo;
    }

    void propagate_factual_states() {
        std::vector<uint32_t> order = get_topological_order();
        for (uint32_t id : order) {
            auto& node = nodes_[id];
            if (node.is_intervened) {
                node.factual_state = node.intervention_value;
            } else {
                std::vector<float> parent_vals;
                for (uint32_t pid : node.parents) {
                    parent_vals.push_back(nodes_[pid].factual_state);
                }
                node.factual_state = node.structural_eq(parent_vals, node.exogenous_noise);
            }
        }
    }

    void apply_do_intervention(uint32_t target_node_id, float value) {
        auto it = nodes_.find(target_node_id);
        if (it != nodes_.end()) {
            it->second.is_intervened = true;
            it->second.intervention_value = value;
            it->second.counterfactual_state = value;
        }
    }

    void reset_interventions() {
        for (auto& pair : nodes_) {
            pair.second.is_intervened = false;
            pair.second.intervention_value = 0.0f;
        }
    }

    void propagate_counterfactual_states() {
        std::vector<uint32_t> order = get_topological_order();
        for (uint32_t id : order) {
            auto& node = nodes_[id];
            if (node.is_intervened) {
                node.counterfactual_state = node.intervention_value;
            } else {
                std::vector<float> parent_vals;
                for (uint32_t pid : node.parents) {
                    parent_vals.push_back(nodes_[pid].counterfactual_state);
                }
                node.counterfactual_state = node.structural_eq(parent_vals, node.exogenous_noise);
            }
        }
    }
};

// =========================================================================
// 3. Counterfactual Observer & Variance Ledger
// =========================================================================
struct CounterfactualTrace {
    uint32_t node_id;
    float factual_val;
    float counterfactual_val;
    float variance; // |Factual - Counterfactual|
};

class CounterfactualObserver {
public:
    static std::vector<CounterfactualTrace> evaluate_intervention_impact(
        StructuralCausalModel& scm,
        uint32_t intervention_node_id,
        float intervention_value)
    {
        scm.propagate_factual_states();
        scm.apply_do_intervention(intervention_node_id, intervention_value);
        scm.propagate_counterfactual_states();

        std::vector<CounterfactualTrace> traces;
        for (const auto& pair : scm.get_all_nodes()) {
            const auto& node = pair.second;
            float var = std::abs(node.factual_state - node.counterfactual_state);
            traces.push_back({
                node.id,
                node.factual_state,
                node.counterfactual_state,
                var
            });
        }
        return traces;
    }
};

class CausalCollapseEngine {
private:
    float collapse_threshold_;

public:
    explicit CausalCollapseEngine(float threshold = 0.2f)
        : collapse_threshold_(threshold) {}

    size_t execute_relationship_collapse(
        StructuralCausalModel& scm,
        const std::vector<CounterfactualTrace>& traces)
    {
        size_t collapsed_count = 0;
        std::unordered_map<uint32_t, float> var_map;
        for (const auto& trace : traces) {
            var_map[trace.node_id] = trace.variance;
        }

        for (const auto& pair : scm.get_all_nodes()) {
            auto* node = scm.get_node(pair.first);
            if (!node) continue;

            float var = var_map[node->id];
            if (var > collapse_threshold_ || node->is_intervened) {
                node->manifested = true;
                collapsed_count++;
            } else {
                node->manifested = false;
            }
        }
        return collapsed_count;
    }
};

// =========================================================================
// 4. Protocol Execution Boundary & Structural Fracture Diagnostics
// =========================================================================
struct ProtocolBoundary {
    std::string protocol_id;
    float min_state_bound = -1.0f;
    float max_state_bound = 100.0f;
    float max_latency_ms = 16.67f;
    uint32_t memory_domain_id = 1;
};

struct BoundaryCollisionEvent {
    std::string protocol_a;
    std::string protocol_b;
    std::string cause_description;
    float value_at_collision;
    bool fracture_detected;
};

class ProtocolBoundaryLedger {
private:
    std::unordered_map<std::string, ProtocolBoundary> boundaries_;

public:
    void register_protocol_boundary(const ProtocolBoundary& boundary) {
        boundaries_[boundary.protocol_id] = boundary;
    }

    const ProtocolBoundary* get_boundary(const std::string& protocol_id) const {
        auto it = boundaries_.find(protocol_id);
        return (it != boundaries_.end()) ? &it->second : nullptr;
    }

    BoundaryCollisionEvent diagnose_boundary_collision(
        const std::string& proto_a_id,
        float proto_a_value,
        const std::string& proto_b_id,
        float proto_b_value,
        float elapsed_time_ms)
    {
        BoundaryCollisionEvent event;
        event.protocol_a = proto_a_id;
        event.protocol_b = proto_b_id;
        event.value_at_collision = proto_a_value;
        event.fracture_detected = false;

        const auto* bound_a = get_boundary(proto_a_id);
        const auto* bound_b = get_boundary(proto_b_id);

        if (!bound_a || !bound_b) {
            event.fracture_detected = true;
            event.cause_description = "Unregistered protocol boundary metadata.";
            return event;
        }

        if (proto_a_value < bound_a->min_state_bound || proto_a_value > bound_a->max_state_bound) {
            event.fracture_detected = true;
            event.cause_description = "Protocol " + proto_a_id + " value (" + std::to_string(proto_a_value) +
                                      ") exceeded state boundary [" + std::to_string(bound_a->min_state_bound) +
                                      ", " + std::to_string(bound_a->max_state_bound) + "]";
            return event;
        }

        if (proto_a_value > bound_b->max_state_bound || proto_b_value < bound_a->min_state_bound) {
            event.fracture_detected = true;
            event.cause_description = "Domain disconnect between Protocol " + proto_a_id + " and " + proto_b_id +
                                      ": boundary ranges do not overlap.";
            return event;
        }

        if (elapsed_time_ms > bound_a->max_latency_ms) {
            event.fracture_detected = true;
            event.cause_description = "Time boundary fracture in Protocol " + proto_a_id +
                                      ": elapsed " + std::to_string(elapsed_time_ms) + "ms > max " +
                                      std::to_string(bound_a->max_latency_ms) + "ms";
            return event;
        }

        event.cause_description = "Protocols operating within aligned boundary intersection.";
        return event;
    }
};

// Backward compatibility alias for CausalGraphManager
using CausalGraphManager = StructuralCausalModel;

} // namespace core
} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP
