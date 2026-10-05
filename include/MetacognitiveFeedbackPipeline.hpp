#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>
#include "FractalCausalTree.hpp"
#include "AutonomicStateController.hpp"

class MetacognitiveFeedbackPipeline {
private:
    float perceived_meta_stress_{0.0f};      // 상향식 메타인지 스트레스 (0.0 ~ 1.0)
    float adaptive_mitosis_threshold_{0.85f}; // 동적 세포분열 임계값

public:
    void process_bottom_up_feedback(
        const FractalCausalTree& tree,
        AutonomicStateController& autonomic_ctrl
    ) {
        float systemic_coherence = tree.calculate_systemic_fractal_coherence();
        perceived_meta_stress_ = 1.0f - std::clamp(systemic_coherence, 0.0f, 1.0f);

        if (perceived_meta_stress_ > 0.60f) {
            adaptive_mitosis_threshold_ = std::max(0.40f, 0.85f - perceived_meta_stress_ * 0.45f);
            std::cout << "[Metacognitive Feedback] High Dissonance Detected! (Meta Stress: "
                      << perceived_meta_stress_ << ")\n"
                      << " -> Triggering Sympathetic Alarm & Lowering Mitosis Threshold: "
                      << adaptive_mitosis_threshold_ << "\n";
        } else {
            adaptive_mitosis_threshold_ = 0.85f;
        }

        autonomic_ctrl.inject_metacognitive_stress(perceived_meta_stress_);
    }

    [[nodiscard]] float get_adaptive_mitosis_threshold() const { return adaptive_mitosis_threshold_; }
    [[nodiscard]] float get_perceived_meta_stress() const { return perceived_meta_stress_; }
};
