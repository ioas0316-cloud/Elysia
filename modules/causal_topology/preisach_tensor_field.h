#ifndef PREISACH_TENSOR_FIELD_H
#define PREISACH_TENSOR_FIELD_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// 1. N-Dimensional Tensor Field Node Preisach Hysteresis Memory (SoA Layout)
// =============================================================================
struct PreisachTensorFieldSoA {
    size_t num_nodes;
    size_t num_hysterons;

    // 1. Current field signals & frozen remanence states
    std::vector<float> input_signals;      // u(t): external intervention / control signal
    std::vector<float> remanence_states;   // S_r(t): frozen remanence state via hysteresis

    // 2. Preisach threshold grid & density weights
    std::vector<float> alpha_grid;         // Switch-ON thresholds
    std::vector<float> beta_grid;          // Switch-OFF thresholds
    std::vector<float> density_weights;    // mu(alpha, beta): causal weights per hysteron

    // 3. Bit-packed Hysteron ON/OFF states (Bitmask array per node)
    std::vector<uint64_t> hysteron_bitmasks;

    // Constructor with default grid initialization
    PreisachTensorFieldSoA(size_t nodes = 64, size_t hysterons_per_dim = 8)
        : num_nodes(nodes), num_hysterons(hysterons_per_dim * hysterons_per_dim) {

        input_signals.resize(num_nodes, 0.0f);
        remanence_states.resize(num_nodes, 0.0f);

        size_t uint64_per_node = (num_hysterons + 63) / 64;
        hysteron_bitmasks.resize(num_nodes * uint64_per_node, 0ULL);

        InitPreisachGrid(1.0f, hysterons_per_dim);
    }

    // Initialize Preisach grid over upper-triangular region (alpha >= beta)
    void InitPreisachGrid(float u_max = 1.0f, size_t steps = 8) {
        alpha_grid.clear();
        beta_grid.clear();
        density_weights.clear();

        if (steps == 0) steps = 1;
        float step_size = (2.0f * u_max) / static_cast<float>(steps);

        for (size_t i = 0; i < steps; ++i) {
            float alpha = -u_max + static_cast<float>(i + 1) * step_size;
            for (size_t j = 0; j < steps; ++j) {
                float beta = -u_max + static_cast<float>(j) * step_size;
                if (alpha >= beta) { // Upper triangular domain constraint
                    alpha_grid.push_back(alpha);
                    beta_grid.push_back(beta);
                    density_weights.push_back(0.5f); // Default uniform density
                }
            }
        }

        num_hysterons = alpha_grid.size();
        size_t uint64_per_node = (num_hysterons + 63) / 64;
        hysteron_bitmasks.assign(num_nodes * uint64_per_node, 0ULL);
        input_signals.assign(num_nodes, 0.0f);
        remanence_states.assign(num_nodes, 0.0f);
    }
};

// =============================================================================
// 2. 1-Step Parallel Pipeline for Preisach Field Update & Remanence Freezing
// =============================================================================
inline void UpdatePreisachTensorField(PreisachTensorFieldSoA& field) {
    const size_t num_nodes = field.num_nodes;
    const size_t num_hysterons = field.num_hysterons;
    if (num_nodes == 0 || num_hysterons == 0) return;

    const size_t uint64_per_node = (num_hysterons + 63) / 64;

    #pragma omp parallel for if(num_nodes > 16)
    for (size_t i = 0; i < num_nodes; ++i) {
        float u_t = field.input_signals[i];
        float accumulated_remanence = 0.0f;

        for (size_t h = 0; h < num_hysterons; ++h) {
            float alpha = field.alpha_grid[h];
            float beta  = field.beta_grid[h];
            float weight = field.density_weights[h];

            size_t mask_array_idx = i * uint64_per_node + (h / 64);
            uint64_t bit_pos = 1ULL << (h % 64);

            // 1. Hysteron State Switching Logic
            bool current_state = (field.hysteron_bitmasks[mask_array_idx] & bit_pos) != 0;

            if (u_t >= alpha) {
                current_state = true;  // Switch ON
            } else if (u_t <= beta) {
                current_state = false; // Switch OFF
            }

            // 2. Bitmask Update & Accumulation
            if (current_state) {
                field.hysteron_bitmasks[mask_array_idx] |= bit_pos;
                accumulated_remanence += weight * 1.0f;
            } else {
                field.hysteron_bitmasks[mask_array_idx] &= ~bit_pos;
                accumulated_remanence += weight * -1.0f;
            }
        }

        // 3. Freeze remanence state S_r(t)
        field.remanence_states[i] = accumulated_remanence;
    }
}

// =============================================================================
// 3. Dynamic Preisach Density Tuning (Condensation vs Unlocking)
// =============================================================================
inline void UpdateCausalTensionAndDensity(
    PreisachTensorFieldSoA& field,
    const std::vector<float>& do_interventions,
    const std::vector<float>& observed_outputs,
    const std::vector<float>& predicted_outputs,
    float dt,
    float eta_cond = 0.15f,
    float eta_unlock = 0.25f,
    float sigma2 = 0.05f)
{
    const size_t num_nodes = field.num_nodes;
    const size_t num_hysterons = field.num_hysterons;
    if (num_nodes == 0 || num_hysterons == 0) return;

    for (size_t i = 0; i < num_nodes; ++i) {
        float do_x = do_interventions[i];
        float y_obs = (i < observed_outputs.size()) ? observed_outputs[i] : 0.0f;
        float y_pred = (i < predicted_outputs.size()) ? predicted_outputs[i] : 0.0f;

        // 1. Tension (Error/Inconsistency)
        float tension = std::abs(y_obs - y_pred);

        // 2. Invariance (Symmetry/Predictability score)
        float invariance = 1.0f / (1.0f + tension * tension);

        // 3. Switching coordinate mapping (symmetric switching model)
        float current_alpha = do_x;
        float current_beta  = -do_x;

        // 4. Update mu density for hysterons
        for (size_t h = 0; h < num_hysterons; ++h) {
            float a = field.alpha_grid[h];
            float b = field.beta_grid[h];
            float current_mu = field.density_weights[h];

            float dist_sq = (a - current_alpha) * (a - current_alpha) + (b - current_beta) * (b - current_beta);
            float spatial_kernel = std::exp(-dist_sq / (2.0f * sigma2));

            float d_mu_cond = eta_cond * invariance * spatial_kernel;
            float d_mu_unlock = -eta_unlock * tension * current_mu * spatial_kernel;

            float next_mu = current_mu + (d_mu_cond + d_mu_unlock) * dt;
            field.density_weights[h] = std::max(0.0f, next_mu);
        }
    }
}

// =============================================================================
// 4. Upper Symbolic Causal Graph Structures & Attractor Extraction Layer
// =============================================================================
struct MacroSymbolNode {
    uint32_t node_id;
    float pivot_alpha;      // Attractor center alpha
    float pivot_beta;       // Attractor center beta
    float axiom_rigidity;   // Density integral / Rigidity degree
    float current_state_sr; // Ensemble mean remanence state
};

struct CausalEdge {
    uint32_t source_node_id;
    uint32_t target_node_id;
    float causal_weight;    // Transition influence / Causal power
    float reluctance;       // Hysteresis impedance / Reluctance
};

class AttractorExtractionLayer {
public:
    void ExtractCausalGraph(
        const PreisachTensorFieldSoA& field,
        std::vector<MacroSymbolNode>& out_nodes,
        std::vector<CausalEdge>& out_edges,
        float density_threshold = 0.4f,
        float max_reluctance = 1.8f)
    {
        out_nodes.clear();
        out_edges.clear();

        if (field.num_hysterons == 0) return;

        // 1. Scan density peaks above threshold -> MacroSymbolNodes
        for (size_t h = 0; h < field.num_hysterons; ++h) {
            float mu = field.density_weights[h];
            if (mu >= density_threshold) {
                MacroSymbolNode node;
                node.node_id = static_cast<uint32_t>(out_nodes.size());
                node.pivot_alpha = field.alpha_grid[h];
                node.pivot_beta = field.beta_grid[h];
                node.axiom_rigidity = mu;

                // Mean remanence S_r across nodes
                float sum_sr = 0.0f;
                if (field.num_nodes > 0) {
                    for (size_t n = 0; n < field.num_nodes; ++n) {
                        sum_sr += field.remanence_states[n];
                    }
                    sum_sr /= static_cast<float>(field.num_nodes);
                }
                node.current_state_sr = sum_sr;

                out_nodes.push_back(node);
            }
        }

        // 2. Derive Directed Edges & Reluctance between MacroSymbolNodes
        for (size_t i = 0; i < out_nodes.size(); ++i) {
            for (size_t j = 0; j < out_nodes.size(); ++j) {
                if (i == j) continue;

                float da = out_nodes[j].pivot_alpha - out_nodes[i].pivot_alpha;
                float db = out_nodes[j].pivot_beta - out_nodes[i].pivot_beta;
                float reluctance = std::sqrt(da * da + db * db);

                if (reluctance <= max_reluctance) {
                    CausalEdge edge;
                    edge.source_node_id = out_nodes[i].node_id;
                    edge.target_node_id = out_nodes[j].node_id;
                    edge.reluctance = reluctance;
                    edge.causal_weight = (out_nodes[i].axiom_rigidity * out_nodes[j].axiom_rigidity) / (1.0f + reluctance);
                    out_edges.push_back(edge);
                }
            }
        }
    }
};

// =============================================================================
// 5. Causal Backward Trajectory Tracing Engine (Modified Backward A*)
// =============================================================================
class CausalBacktracer {
public:
    std::vector<uint32_t> TraceMinimalImpedancePath(
        uint32_t goal_node_id,
        uint32_t start_node_id,
        const std::vector<MacroSymbolNode>& nodes,
        const std::vector<CausalEdge>& edges)
    {
        if (nodes.empty()) return {};
        if (start_node_id >= nodes.size() || goal_node_id >= nodes.size()) return {};
        if (start_node_id == goal_node_id) return {start_node_id};

        // Build In-Edge map (target -> list of in-edges from sources)
        std::unordered_map<uint32_t, std::vector<CausalEdge>> in_edges_map;
        for (const auto& edge : edges) {
            in_edges_map[edge.target_node_id].push_back(edge);
        }

        using PQItem = std::pair<float, uint32_t>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> open_set;

        std::unordered_map<uint32_t, float> g_score;
        std::unordered_map<uint32_t, uint32_t> parent_map;

        g_score[goal_node_id] = 0.0f;
        open_set.push({0.0f, goal_node_id});

        const float lambda_rigidity = 0.4f;
        const float epsilon = 0.001f;

        while (!open_set.empty()) {
            auto [current_cost, curr_node] = open_set.top();
            open_set.pop();

            if (curr_node == start_node_id) {
                break;
            }

            if (current_cost > g_score[curr_node]) continue;

            auto it = in_edges_map.find(curr_node);
            if (it != in_edges_map.end()) {
                for (const auto& edge : it->second) {
                    uint32_t prev_node = edge.source_node_id;
                    float prev_rigidity = (prev_node < nodes.size()) ? nodes[prev_node].axiom_rigidity : 0.5f;

                    float transition_cost = edge.reluctance / (edge.causal_weight + epsilon);
                    float rigidity_penalty = lambda_rigidity * (1.0f - prev_rigidity);
                    float edge_cost = transition_cost + rigidity_penalty;

                    float new_g_score = g_score[curr_node] + edge_cost;

                    if (g_score.find(prev_node) == g_score.end() || new_g_score < g_score[prev_node]) {
                        g_score[prev_node] = new_g_score;
                        parent_map[prev_node] = curr_node; // Reverse link (prev -> curr)
                        open_set.push({new_g_score, prev_node});
                    }
                }
            }
        }

        if (g_score.find(start_node_id) == g_score.end()) {
            // Fallback direct path if no graph connection found
            return {start_node_id, goal_node_id};
        }

        // Reconstruct forward path from start to goal
        std::vector<uint32_t> causal_trajectory;
        uint32_t curr = start_node_id;
        while (curr != goal_node_id) {
            causal_trajectory.push_back(curr);
            auto p_it = parent_map.find(curr);
            if (p_it == parent_map.end()) break;
            curr = p_it->second;
        }
        causal_trajectory.push_back(goal_node_id);

        return causal_trajectory;
    }
};

// =============================================================================
// 6. Bi-directional Closed-Loop Feedback Control System
// =============================================================================
class ClosedLoopCausalEngine {
public:
    bool ExecuteAndAdaptTrajectory(
        const std::vector<uint32_t>& trajectory,
        const std::vector<MacroSymbolNode>& symbol_nodes,
        PreisachTensorFieldSoA& lower_field,
        float tension_threshold = 0.20f,
        float dt = 0.01f)
    {
        bool feedback_triggered = false;

        for (uint32_t node_id : trajectory) {
            if (node_id >= symbol_nodes.size()) continue;
            const auto& symbol = symbol_nodes[node_id];

            float current_field_sr = 0.0f;
            if (lower_field.num_nodes > 0) {
                for (size_t i = 0; i < lower_field.num_nodes; ++i) {
                    current_field_sr += lower_field.remanence_states[i];
                }
                current_field_sr /= static_cast<float>(lower_field.num_nodes);
            }

            float tension = std::abs(current_field_sr - symbol.current_state_sr);

            if (tension > tension_threshold) {
                feedback_triggered = true;

                // Inject top-down degaussing pulse u(t) into lower field
                InjectTopDownIntervention(lower_field, symbol.pivot_alpha, symbol.pivot_beta, tension, dt);

                // Phase Unlocking: soften density weights in high tension region
                for (size_t h = 0; h < lower_field.num_hysterons; ++h) {
                    float da = lower_field.alpha_grid[h] - symbol.pivot_alpha;
                    float db = lower_field.beta_grid[h] - symbol.pivot_beta;
                    if ((da * da + db * db) < 0.2f) {
                        lower_field.density_weights[h] *= 0.8f; // Dynamic softening
                    }
                }
            }
        }

        return feedback_triggered;
    }

private:
    void InjectTopDownIntervention(
        PreisachTensorFieldSoA& field,
        float target_alpha,
        float target_beta,
        float tension,
        float dt)
    {
        float amplitude = tension * 1.5f;
        float frequency = 50.0f;

        #pragma omp parallel for if(field.num_nodes > 16)
        for (size_t i = 0; i < field.num_nodes; ++i) {
            float dist = std::abs(field.input_signals[i] - target_alpha);
            float spatial_decay = std::exp(-dist * dist / 0.1f);
            float u_pulse = amplitude * spatial_decay * std::cos(frequency * dt);

            field.input_signals[i] += u_pulse;
        }

        UpdatePreisachTensorField(field);
    }
};

#endif // PREISACH_TENSOR_FIELD_H
