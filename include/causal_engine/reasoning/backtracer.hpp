#ifndef CAUSAL_ENGINE_REASONING_BACKTRACER_HPP
#define CAUSAL_ENGINE_REASONING_BACKTRACER_HPP

#include "causal_engine/core/preisach_soa.hpp"

// Enhanced CausalBacktracer with Curvature & Multi-Scale Latency Damping
class EnhancedCausalBacktracer : public CausalBacktracer {
public:
    std::vector<uint32_t> TraceMinimalImpedancePathWithLatency(
        uint32_t goal_node_id,
        uint32_t start_node_id,
        const std::vector<MacroSymbolNode>& nodes,
        const std::vector<CausalEdge>& edges,
        float gamma_curvature = 0.2f,
        float latency_damping = 0.1f)
    {
        if (nodes.empty()) return {};
        if (start_node_id >= nodes.size() || goal_node_id >= nodes.size()) return {};
        if (start_node_id == goal_node_id) return {start_node_id};

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

            if (curr_node == start_node_id) break;
            if (current_cost > g_score[curr_node]) continue;

            auto it = in_edges_map.find(curr_node);
            if (it != in_edges_map.end()) {
                for (const auto& edge : it->second) {
                    uint32_t prev_node = edge.source_node_id;
                    float prev_rigidity = (prev_node < nodes.size()) ? nodes[prev_node].axiom_rigidity : 0.5f;

                    // Transition cost & rigidity penalty
                    float transition_cost = edge.reluctance / (edge.causal_weight + epsilon);
                    float rigidity_penalty = lambda_rigidity * (1.0f - prev_rigidity);

                    // Trajectory Curvature Angle calculation
                    float curvature_cost = 0.0f;
                    if (parent_map.find(curr_node) != parent_map.end()) {
                        uint32_t next_node = parent_map[curr_node];
                        if (prev_node < nodes.size() && curr_node < nodes.size() && next_node < nodes.size()) {
                            float v1_a = nodes[curr_node].pivot_alpha - nodes[prev_node].pivot_alpha;
                            float v1_b = nodes[curr_node].pivot_beta - nodes[prev_node].pivot_beta;
                            float v2_a = nodes[next_node].pivot_alpha - nodes[curr_node].pivot_alpha;
                            float v2_b = nodes[next_node].pivot_beta - nodes[curr_node].pivot_beta;

                            float norm1 = std::sqrt(v1_a * v1_a + v1_b * v1_b) + epsilon;
                            float norm2 = std::sqrt(v2_a * v2_a + v2_b * v2_b) + epsilon;
                            float dot = (v1_a * v2_a + v1_b * v2_b) / (norm1 * norm2);
                            dot = std::clamp(dot, -1.0f, 1.0f);

                            // Curvature angle penalty (1 - cos(theta))
                            curvature_cost = gamma_curvature * (1.0f - dot);
                        }
                    }

                    // Multi-scale latency damping factor
                    float multi_scale_damping = 1.0f / (1.0f + latency_damping * current_cost);

                    float edge_cost = (transition_cost + rigidity_penalty + curvature_cost) * multi_scale_damping;
                    float new_g_score = g_score[curr_node] + edge_cost;

                    if (g_score.find(prev_node) == g_score.end() || new_g_score < g_score[prev_node]) {
                        g_score[prev_node] = new_g_score;
                        parent_map[prev_node] = curr_node;
                        open_set.push({new_g_score, prev_node});
                    }
                }
            }
        }

        if (g_score.find(start_node_id) == g_score.end()) {
            return {start_node_id, goal_node_id};
        }

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

#endif // CAUSAL_ENGINE_REASONING_BACKTRACER_HPP
