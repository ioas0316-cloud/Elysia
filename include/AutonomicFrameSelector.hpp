#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include "CausalDataNode.hpp"
#include "CausalMeaningEvaluator.hpp"
#include "AutonomicStateController.hpp"

class AutonomicFrameSelector {
private:
    float minimum_required_resolution_{0.80f}; // 최소 요구 인과 해상도 (80%)

public:
    AutonomicFrameSelector(float min_res = 0.80f) : minimum_required_resolution_(min_res) {}

    void auto_select_optimal_frame(
        CausalDataNode& data_node,
        const std::vector<MeaningInterpretation>& evaluations,
        AutonomicStateController& autonomic_ctrl
    ) {
        const MeaningInterpretation* best_frame = nullptr;
        float max_resolution = -1.0f;

        for (const auto& eval : evaluations) {
            if (eval.is_valid_reading && eval.resolvable_degree > max_resolution) {
                max_resolution = eval.resolvable_degree;
                best_frame = &eval;
            }
        }

        if (best_frame && max_resolution >= minimum_required_resolution_) {
            std::cout << "[AutonomicFrameSelector] Optimal Observational Frame Selected: "
                      << best_frame->frame_name << " (Resolution: " << (max_resolution * 100.0f) << "%)\n";

            int target_space = 0;
            if (best_frame->frame_name.find("Complex") != std::string::npos) {
                target_space = 1;
            } else if (best_frame->frame_name.find("Clifford") != std::string::npos) {
                target_space = 2;
            }

            data_node.shift_observational_frame(make_float2(0.7071f, 0.7071f), 0.5f, target_space);
            autonomic_ctrl.inject_metacognitive_stress(0.1f);
        } else {
            std::cout << "[AutonomicFrameSelector] WARNING: No Suitable Frame Met Target Resolution!\n"
                      << " -> Triggering Sympathetic Mitotic Escalation...\n";
            autonomic_ctrl.inject_metacognitive_stress(0.95f);
        }
    }
};
