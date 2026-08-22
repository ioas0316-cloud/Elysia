#ifndef CAUSAL_ENGINE_FEEDBACK_CAUSAL_IMPEDANCE_HPP
#define CAUSAL_ENGINE_FEEDBACK_CAUSAL_IMPEDANCE_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <string>
#include <iostream>
#include "../core/preisach_soa.hpp"

namespace causal_engine {

// =============================================================================
// 1. Structural Causal Impedance Metrics (Curvature, Phase Discrepancy, Friction)
// =============================================================================

struct ImpedanceResult {
    float trajectory_curvature;       // 궤적 꺾임 (Curvature angle deviation)
    float topological_phase_diff;     // 위상 불일치 (Topological Phase Discrepancy)
    float latency_damped_friction;    // 느린 상위 댐퍼가 적용된 정보적 마찰 (Impedance)
    float resonance_score;            // 공명 점수 (1 / (1 + friction))
    bool requires_rule_mutation;       // 규칙 변형(Rule Mutation) 유발 여부
};

class CausalImpedanceEvaluator {
public:
    // Compute trajectory curvature across candidate sequence of pivot coordinates (alpha, beta)
    static float ComputeTrajectoryCurvature(const std::vector<MacroSymbolNode>& nodes, const std::vector<uint32_t>& trajectory) {
        if (trajectory.size() < 3) return 0.0f;

        float total_curvature = 0.0f;
        size_t count = 0;

        for (size_t i = 1; i < trajectory.size() - 1; ++i) {
            uint32_t prev_idx = trajectory[i - 1];
            uint32_t curr_idx = trajectory[i];
            uint32_t next_idx = trajectory[i + 1];

            if (prev_idx >= nodes.size() || curr_idx >= nodes.size() || next_idx >= nodes.size()) continue;

            const auto& p1 = nodes[prev_idx];
            const auto& p2 = nodes[curr_idx];
            const auto& p3 = nodes[next_idx];

            // Vectors: v1 = p2 - p1, v2 = p3 - p2
            float v1_a = p2.pivot_alpha - p1.pivot_alpha;
            float v1_b = p2.pivot_beta - p1.pivot_beta;
            float v2_a = p3.pivot_alpha - p2.pivot_alpha;
            float v2_b = p3.pivot_beta - p2.pivot_beta;

            float norm1 = std::sqrt(v1_a * v1_a + v1_b * v1_b);
            float norm2 = std::sqrt(v2_a * v2_a + v2_b * v2_b);

            if (norm1 > 1e-6f && norm2 > 1e-6f) {
                float dot = (v1_a * v2_a + v1_b * v2_b) / (norm1 * norm2);
                dot = std::clamp(dot, -1.0f, 1.0f);
                float angle = std::acos(dot); // Curvature angle in radians [0, pi]
                total_curvature += angle;
                count++;
            }
        }

        return (count > 0) ? (total_curvature / static_cast<float>(count)) : 0.0f;
    }

    // Compute topological phase discrepancy between candidate trajectory and target macro trajectory/concept
    static float ComputeTopologicalPhaseDiscrepancy(
        const std::vector<MacroSymbolNode>& nodes,
        const std::vector<uint32_t>& candidate_trajectory,
        const std::vector<uint32_t>& target_trajectory)
    {
        if (candidate_trajectory.empty() || target_trajectory.empty()) return 1.0f;

        // Compare sequence length ratio discrepancy
        float len_ratio = std::abs(static_cast<float>(candidate_trajectory.size()) - static_cast<float>(target_trajectory.size()))
                          / std::max(candidate_trajectory.size(), target_trajectory.size());

        // Compare endpoint & centroid distance in pivot space
        float cand_a = 0.0f, cand_b = 0.0f;
        for (uint32_t idx : candidate_trajectory) {
            if (idx < nodes.size()) {
                cand_a += nodes[idx].pivot_alpha;
                cand_b += nodes[idx].pivot_beta;
            }
        }
        cand_a /= candidate_trajectory.size();
        cand_b /= candidate_trajectory.size();

        float tgt_a = 0.0f, tgt_b = 0.0f;
        for (uint32_t idx : target_trajectory) {
            if (idx < nodes.size()) {
                tgt_a += nodes[idx].pivot_alpha;
                tgt_b += nodes[idx].pivot_beta;
            }
        }
        tgt_a /= target_trajectory.size();
        tgt_b /= target_trajectory.size();

        float centroid_dist = std::sqrt((cand_a - tgt_a) * (cand_a - tgt_a) + (cand_b - tgt_b) * (cand_b - tgt_b));

        return 0.5f * len_ratio + 0.5f * (1.0f - std::exp(-centroid_dist));
    }

    // Full Impedance Evaluation with Slow Scale Latency Damping
    static ImpedanceResult EvaluateImpedance(
        const std::vector<MacroSymbolNode>& nodes,
        const std::vector<uint32_t>& candidate_trajectory,
        const std::vector<uint32_t>& target_trajectory,
        float gamma_curvature = 0.3f,
        float latency_damping = 0.2f,
        float friction_threshold = 0.45f)
    {
        float curvature = ComputeTrajectoryCurvature(nodes, candidate_trajectory);
        float phase_diff = ComputeTopologicalPhaseDiscrepancy(nodes, candidate_trajectory, target_trajectory);

        // Raw impedance calculation: curvature + phase_diff
        float raw_friction = gamma_curvature * curvature + (1.0f - gamma_curvature) * phase_diff;

        // Apply slow scale latency damping: slow rotor filters out high-frequency noise
        float damped_friction = raw_friction * (1.0f - latency_damping);

        float resonance = 1.0f / (1.0f + damped_friction);
        bool mutate_rule = (damped_friction > friction_threshold);

        return ImpedanceResult{
            curvature,
            phase_diff,
            damped_friction,
            resonance,
            mutate_rule
        };
    }
};

// =============================================================================
// 2. Meta-Constraint Engine & Dynamic Rule Mutation (Meta-Feedback Loop)
// =============================================================================

struct MetaConstraintRule {
    float max_reluctance_threshold; // 연결 허용 최대 이력 저항 (Reluctance)
    float min_rigidity_threshold;   // 노드 생존 최소 강직성 (Axiom Rigidity)
    float alpha_boundary_min;       // 허용 위상 공간 Alpha 하한선
    float alpha_boundary_max;       // 허용 위상 공간 Alpha 상한선
    float beta_boundary_min;        // 허용 위상 공간 Beta 하한선
    float beta_boundary_max;        // 허용 위상 공간 Beta 상한선
    float curvature_penalty_weight; // 궤적 꺾임에 부여하는 구속 가중치

    MetaConstraintRule()
        : max_reluctance_threshold(1.8f),
          min_rigidity_threshold(0.2f),
          alpha_boundary_min(-1.0f),
          alpha_boundary_max(1.0f),
          beta_boundary_min(-1.0f),
          beta_boundary_max(1.0f),
          curvature_penalty_weight(0.5f) {}
};

class MetaConstraintMutator {
private:
    MetaConstraintRule current_rule;
    size_t mutation_count;
    float accumulated_friction;

public:
    MetaConstraintMutator() : mutation_count(0), accumulated_friction(0.0f) {}

    const MetaConstraintRule& GetCurrentRule() const { return current_rule; }
    size_t GetMutationCount() const { return mutation_count; }

    // Mutate Meta-Constraint Rule (Rule Mutation: Constraint A -> Constraint A') based on observed impedance
    void MutateRule(const ImpedanceResult& impedance, const std::vector<MacroSymbolNode>& nodes, const std::vector<uint32_t>& trajectory) {
        mutation_count++;
        accumulated_friction += impedance.latency_damped_friction;

        // 1. If trajectory curvature is high, tighten reluctance threshold & increase curvature penalty
        if (impedance.trajectory_curvature > 0.5f) {
            current_rule.max_reluctance_threshold *= 0.85f; // Reject high reluctance edges causing sharp bends
            current_rule.curvature_penalty_weight += 0.1f;
        }

        // 2. If phase discrepancy is high, adjust boundary limits around target centroid
        if (impedance.topological_phase_diff > 0.4f && !nodes.empty() && !trajectory.empty()) {
            float mean_a = 0.0f, mean_b = 0.0f;
            size_t valid = 0;
            for (uint32_t idx : trajectory) {
                if (idx < nodes.size()) {
                    mean_a += nodes[idx].pivot_alpha;
                    mean_b += nodes[idx].pivot_beta;
                    valid++;
                }
            }
            if (valid > 0) {
                mean_a /= valid;
                mean_b /= valid;

                // Shift and contract state space boundaries toward effective causal region
                current_rule.alpha_boundary_min = std::max(-1.0f, mean_a - 0.6f);
                current_rule.alpha_boundary_max = std::min(1.0f, mean_a + 0.6f);
                current_rule.beta_boundary_min  = std::max(-1.0f, mean_b - 0.6f);
                current_rule.beta_boundary_max  = std::min(1.0f, mean_b + 0.6f);
            }
        }

        // 3. Elevate minimum rigidity threshold to prune weak/noisy attractors
        current_rule.min_rigidity_threshold = std::min(0.8f, current_rule.min_rigidity_threshold + 0.05f);
    }

    // Filter symbol nodes according to current Meta-Constraint Rule
    std::vector<MacroSymbolNode> FilterNodes(const std::vector<MacroSymbolNode>& input_nodes) const {
        std::vector<MacroSymbolNode> filtered;
        for (const auto& node : input_nodes) {
            if (node.axiom_rigidity >= current_rule.min_rigidity_threshold &&
                node.pivot_alpha >= current_rule.alpha_boundary_min &&
                node.pivot_alpha <= current_rule.alpha_boundary_max &&
                node.pivot_beta  >= current_rule.beta_boundary_min &&
                node.pivot_beta  <= current_rule.beta_boundary_max)
            {
                filtered.push_back(node);
            }
        }
        return filtered;
    }

    // Filter edges according to current Meta-Constraint Rule
    std::vector<CausalEdge> FilterEdges(const std::vector<CausalEdge>& input_edges) const {
        std::vector<CausalEdge> filtered;
        for (const auto& edge : input_edges) {
            if (edge.reluctance <= current_rule.max_reluctance_threshold) {
                filtered.push_back(edge);
            }
        }
        return filtered;
    }
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_FEEDBACK_CAUSAL_IMPEDANCE_HPP
