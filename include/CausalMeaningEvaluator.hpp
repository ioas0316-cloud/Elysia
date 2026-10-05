#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "CausalDataNode.hpp"

struct MeaningInterpretation {
    std::string frame_name;
    bool        is_valid_reading;  // 해당 프레임에서 데이터가 깨지지 않고 해석되는가?
    float       resolvable_degree; // 데이터의 인과적 선명도/해상도 (0.0 ~ 1.0)
    std::string extracted_meaning; // 관측을 통해 추출된 인과 맥락
};

class CausalMeaningEvaluator {
public:
    static std::vector<MeaningInterpretation> evaluate_data_across_frames(
        const CausalDataNode& data_node,
        float raw_value,       // 무정형 수치 데이터
        bool  has_singularity  // 특이점(NaN/Inf) 포함 여부
    ) {
        std::vector<MeaningInterpretation> results;

        if (has_singularity) {
            results.push_back({
                "Real 1D (ℝ)",
                false,
                0.0f,
                "[Crash] 1차원 선형 축에서는 데이터의 특이점을 포섭할 수 없어 파괴됨"
            });
        } else {
            results.push_back({
                "Real 1D (ℝ)",
                true,
                0.33f,
                "[Linear] 데이터의 단순 크기와 단방향 선형 인과 관계만 읽어냄"
            });
        }

        if (has_singularity) {
            results.push_back({
                "Complex 2D (ℂ)",
                true,
                0.66f,
                "[Orbital] 특이점을 회전 위상(e12)으로 우회하여 데이터의 주기성을 포섭함"
            });
        } else {
            results.push_back({
                "Complex 2D (ℂ)",
                true,
                0.66f,
                "[Rotation] 데이터의 위상 변화와 파동적 순환 관계를 읽어냄"
            });
        }

        results.push_back({
            "Clifford 3D (Cℓ₃,₀)",
            true,
            1.00f,
            "[Volumetric Spin] 특이점을 차원 상전이 마디로 받아들여, 데이터의 입체적 스케일 및 다차원 맥락을 완벽히 읽어냄"
        });

        return results;
    }

    static void print_evaluation_report(
        const CausalDataNode& node,
        const std::vector<MeaningInterpretation>& reports
    ) {
        std::cout << "\n=======================================================\n";
        std::cout << " [CausalMeaningEvaluator] Multi-Frame Data Interpretation\n";
        std::cout << " Concept ID: " << node.get_meaning().concept_id
                  << " | Tag: " << node.get_meaning().causal_relation_tag << "\n";
        std::cout << "=======================================================\n";

        for (const auto& r : reports) {
            std::cout << " [Frame Tool: " << r.frame_name << "]\n"
                      << "  - Status: " << (r.is_valid_reading ? "VALID" : "FAILED") << "\n"
                      << "  - Resolvable Degree: " << (r.resolvable_degree * 100.0f) << "%\n"
                      << "  - Extracted Meaning: " << r.extracted_meaning << "\n"
                      << " ------------------------------------------------------\n";
        }
    }
};
