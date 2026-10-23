#ifndef CAUSAL_ENGINE_CORE_META_CAUSAL_MAP_HPP
#define CAUSAL_ENGINE_CORE_META_CAUSAL_MAP_HPP

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace causal_engine {

/**
 * @brief Represents a dynamic Mechanism Node whose identity is defined by its internal
 * dynamic laws, constraints, and structural parameters, rather than a static data value.
 */
class MechanismNode {
public:
    std::string id;
    std::string type_name;
    std::vector<float> state;
    std::vector<float> parameters;
    std::vector<float> parameter_deltas;
    float residual_energy{0.0f};

    MechanismNode(std::string node_id, std::string type, size_t state_dim = 4, size_t param_dim = 4)
        : id(std::move(node_id)), type_name(std::move(type)), state(state_dim, 0.0f),
          parameters(param_dim, 1.0f), parameter_deltas(param_dim, 0.0f) {}

    virtual ~MechanismNode() = default;

    /**
     * @brief Evaluates current constraint violation or residual energy of this mechanism.
     */
    virtual float evaluate_residual() = 0;

    /**
     * @brief Computes parameter deltas to relax internal structural constraints towards equilibrium.
     */
    virtual void project_structural_relaxation(float learning_rate) = 0;

    /**
     * @brief Applies accumulated deltas to update parameters and resets deltas.
     */
    virtual void apply_deltas() {
        for (size_t i = 0; i < parameters.size() && i < parameter_deltas.size(); ++i) {
            parameters[i] += parameter_deltas[i];
            parameter_deltas[i] = 0.0f;
        }
    }

    /**
     * @brief Executes one step of intrinsic dynamics under current parameters.
     */
    virtual void execute_dynamics(float dt) = 0;
};

/**
 * @brief Concrete mechanism enforcing a differential bound between state vectors.
 */
class DifferentialBoundMechanism : public MechanismNode {
public:
    float target_max_diff{1.0f};

    DifferentialBoundMechanism(std::string node_id, float max_diff)
        : MechanismNode(std::move(node_id), "DifferentialBound", 2, 2), target_max_diff(max_diff) {
        parameters[0] = max_diff; // Max allowed difference
        parameters[1] = 0.5f;     // Relaxation stiffness
    }

    float evaluate_residual() override {
        float diff = std::abs(state[0] - state[1]);
        float err = std::max(0.0f, diff - parameters[0]);
        residual_energy = err * err;
        return residual_energy;
    }

    void project_structural_relaxation(float learning_rate) override {
        float diff = std::abs(state[0] - state[1]);
        float err = diff - parameters[0];
        if (err > 1e-4f) {
            float stiffness = parameters[1];
            float correction = err * 0.5f * stiffness * learning_rate;
            if (state[0] > state[1]) {
                state[0] -= correction;
                state[1] += correction;
            } else {
                state[0] += correction;
                state[1] -= correction;
            }
        }
    }

    void execute_dynamics(float dt) override {
        // Dynamic drift towards equilibrium
        evaluate_residual();
    }
};

/**
 * @brief Concrete mechanism enforcing conservation or harmonic potential balance.
 */
class HarmonicConservationMechanism : public MechanismNode {
public:
    HarmonicConservationMechanism(std::string node_id, float target_sum)
        : MechanismNode(std::move(node_id), "HarmonicConservation", 3, 2) {
        parameters[0] = target_sum; // Conservation invariant constant
        parameters[1] = 0.1f;       // Damping coefficient
    }

    float evaluate_residual() override {
        float current_sum = state[0] + state[1] + state[2];
        float err = current_sum - parameters[0];
        residual_energy = 0.5f * err * err;
        return residual_energy;
    }

    void project_structural_relaxation(float learning_rate) override {
        float current_sum = state[0] + state[1] + state[2];
        float err = current_sum - parameters[0];
        if (std::abs(err) > 1e-4f) {
            float shift = (err / 3.0f) * learning_rate;
            state[0] -= shift;
            state[1] -= shift;
            state[2] -= shift;
        }
    }

    void execute_dynamics(float dt) override {
        // Natural state oscillation and damping
        float damping = parameters[1];
        for (auto& s : state) {
            s *= (1.0f - damping * dt);
        }
        evaluate_residual();
    }
};

/**
 * @brief Concrete mechanism representing compressed conceptual grounding (Symbolic Intuition).
 * Evaluates whether two compressed concept mechanisms cancel out or align to equilibrium (x + y = 0).
 */
class SymbolicIntuitionMechanism : public MechanismNode {
public:
    std::string word_a;
    std::string word_b;

    SymbolicIntuitionMechanism(std::string node_id, std::string concept_a, std::string concept_b)
        : MechanismNode(std::move(node_id), "SymbolicIntuition", 2, 2),
          word_a(std::move(concept_a)), word_b(std::move(concept_b)) {
        parameters[0] = 0.0f; // Equilibrium target (0.0 = grounded cancellation)
        parameters[1] = 1.0f; // Intuitive grounding stiffness
    }

    float evaluate_residual() override {
        float equilibrium_diff = state[0] + state[1] - parameters[0];
        residual_energy = 0.5f * equilibrium_diff * equilibrium_diff;
        return residual_energy;
    }

    void project_structural_relaxation(float learning_rate) override {
        float err = state[0] + state[1] - parameters[0];
        if (std::abs(err) > 1e-4f) {
            float shift = (err * 0.5f) * parameters[1] * learning_rate;
            state[0] -= shift;
            state[1] -= shift;
        }
    }

    void execute_dynamics(float dt) override {
        evaluate_residual();
    }
};

/**
 * @brief Coupling edge between mechanisms (Meta-Causality).
 * A change in source mechanism's residual or state reconfigures target mechanism's parameters.
 */
struct CausalBinding {
    std::string source_id;
    std::string target_id;
    float coupling_weight{1.0f};

    // Reconfigures target mechanism based on source mechanism's residual / state
    std::function<void(const MechanismNode& source, MechanismNode& target, float weight)> reconfigure_fn;
};

/**
 * @brief Meta-Causal Engine governing an ecosystem of interconnected mechanisms.
 */
class MetaCausalEngine {
private:
    std::unordered_map<std::string, std::shared_ptr<MechanismNode>> mechanisms_;
    std::vector<std::string> mechanism_order_;
    std::vector<CausalBinding> bindings_;

public:
    MetaCausalEngine() = default;

    void add_mechanism(std::shared_ptr<MechanismNode> node) {
        if (!node) return;
        mechanisms_[node->id] = node;
        mechanism_order_.push_back(node->id);
    }

    std::shared_ptr<MechanismNode> get_mechanism(const std::string& id) const {
        auto it = mechanisms_.find(id);
        if (it != mechanisms_.end()) return it->second;
        return nullptr;
    }

    const std::unordered_map<std::string, std::shared_ptr<MechanismNode>>& get_all_mechanisms() const {
        return mechanisms_;
    }

    void add_binding(CausalBinding binding) {
        bindings_.push_back(std::move(binding));
    }

    void add_binding(const std::string& source_id, const std::string& target_id, float coupling_weight) {
        CausalBinding b;
        b.source_id = source_id;
        b.target_id = target_id;
        b.coupling_weight = coupling_weight;
        b.reconfigure_fn = [](const MechanismNode& source, MechanismNode& target, float weight) {
            // Default meta-causal coupling: source residual modulates target's parameter[0]
            if (!target.parameters.empty()) {
                float shift = source.residual_energy * weight * 0.05f;
                target.parameters[0] = std::max(0.01f, target.parameters[0] + shift);
            }
        };
        bindings_.push_back(std::move(b));
    }

    /**
     * @brief Computes total residual energy across all mechanisms in the ecosystem.
     */
    float compute_total_residual() {
        float total = 0.0f;
        for (auto& pair : mechanisms_) {
            total += pair.second->evaluate_residual();
        }
        return total;
    }

    /**
     * @brief Propagates meta-causal reconfigurations across coupled mechanisms.
     */
    void propagate_meta_causal_bindings() {
        for (const auto& binding : bindings_) {
            auto src_it = mechanisms_.find(binding.source_id);
            auto tgt_it = mechanisms_.find(binding.target_id);
            if (src_it != mechanisms_.end() && tgt_it != mechanisms_.end()) {
                if (binding.reconfigure_fn) {
                    binding.reconfigure_fn(*(src_it->second), *(tgt_it->second), binding.coupling_weight);
                }
            }
        }
    }

    /**
     * @brief Executes autonomous convergence relaxation on the mechanism graph.
     */
    int step_convergence(int max_iterations = 50, float tolerance = 1e-4f, float learning_rate = 0.5f) {
        int iter_count = 0;
        for (int iter = 0; iter < max_iterations; ++iter) {
            iter_count++;

            // 1. Meta-causal propagation: Mechanism states reconfigure connected mechanism parameters
            propagate_meta_causal_bindings();

            // 2. Evaluate total residual energy
            float total_residual = compute_total_residual();
            if (total_residual < tolerance) {
                break;
            }

            // 3. Perform structural relaxation across each mechanism node
            for (const auto& id : mechanism_order_) {
                auto node = mechanisms_[id];
                node->project_structural_relaxation(learning_rate);
                node->apply_deltas();
            }
        }
        return iter_count;
    }

    /**
     * @brief Introspection & Inverse Causal Exploration:
     * Traces residual energy and structural contribution per mechanism to identify
     * which mechanism node drove the system's equilibrium state shift.
     */
    std::unordered_map<std::string, float> introspect_causal_contributions() {
        std::unordered_map<std::string, float> contributions;
        float total_energy = compute_total_residual() + 1e-8f;

        for (const auto& pair : mechanisms_) {
            float node_residual = pair.second->evaluate_residual();
            contributions[pair.first] = node_residual / total_energy;
        }

        return contributions;
    }
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_META_CAUSAL_MAP_HPP
