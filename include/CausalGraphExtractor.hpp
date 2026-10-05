#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "CausalPhaseTensor.hpp"

struct CausalNode {
    int index;               // 텐서 내 마디 인덱스
    float curvature;         // 공간 곡률 (1.0 = 특이점 우회 발생 지점)
    float2 rotor_state;      // 회전 우회 후의 바이브이그저 위상 (cos θ, sin θ)
    std::string phase_space; // 해당 마디가 속했던 대수 공간 상태
};

class CausalGraphExtractor {
public:
    static std::vector<CausalNode> extract_topological_nodes(
        float2* d_rotors,
        float* d_curvature_map,
        size_t num_elements,
        PhaseSpace current_space
    ) {
        std::vector<float> h_curvature(num_elements);
        std::vector<float2> h_rotors(num_elements);

        cudaMemcpy(h_curvature.data(), d_curvature_map, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rotors.data(), d_rotors, num_elements * sizeof(float2), cudaMemcpyDeviceToHost);

        std::vector<CausalNode> extracted_nodes;

        std::string space_str;
        switch (current_space) {
            case PhaseSpace::Real_1D:     space_str = "Real_1D (ℝ)"; break;
            case PhaseSpace::Complex_2D:  space_str = "Complex_2D (ℂ)"; break;
            case PhaseSpace::Clifford_3D: space_str = "Clifford_3D (Cℓ₃,₀)"; break;
        }

        for (size_t i = 0; i < num_elements; ++i) {
            if (h_curvature[i] > 0.5f) {
                extracted_nodes.push_back(CausalNode{
                    static_cast<int>(i),
                    h_curvature[i],
                    h_rotors[i],
                    space_str
                });
            }
        }

        return extracted_nodes;
    }

    static void print_causal_graph_summary(const std::vector<CausalNode>& nodes) {
        std::cout << "\n=====================================================\n";
        std::cout << " [CausalGraphExtractor] Topological Singularity Nodes\n";
        std::cout << "=====================================================\n";
        std::cout << " Total Bypassed Singularities Captured: " << nodes.size() << "\n\n";

        size_t display_limit = std::min(nodes.size(), size_t(5));
        for (size_t i = 0; i < display_limit; ++i) {
            const auto& n = nodes[i];
            float phase_angle = std::atan2(n.rotor_state.y, n.rotor_state.x);
            std::cout << " Node [" << n.index << "] | Space: " << n.phase_space
                      << " | Curvature: " << n.curvature
                      << " | Bypassed Rotor Angle: " << phase_angle << " rad\n";
        }
        if (nodes.size() > display_limit) {
            std::cout << " ... and " << (nodes.size() - display_limit) << " more topological nodes.\n";
        }
        std::cout << "=====================================================\n\n";
    }
};
