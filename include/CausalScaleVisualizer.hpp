#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include "FractalCausalTree.hpp"
#include "CausalPhaseTensor.hpp"

class CausalScaleVisualizer {
public:
    static std::string export_to_json_graph(
        const FractalCausalTree& tree,
        const PhaseSpace current_phase_space
    ) {
        std::ostringstream json;
        json << "{\n";
        json << "  \"meta\": {\n";
        json << "    \"active_phase_space\": \"" << phase_space_to_string(current_phase_space) << "\",\n";
        json << "    \"systemic_coherence\": " << tree.calculate_systemic_fractal_coherence() << "\n";
        json << "  },\n";

        json << "  \"nodes\": [\n";
        const auto& node_map = tree.get_node_table();
        size_t node_idx = 0;

        for (const auto& [id, node] : node_map) {
            float phase_angle = std::atan2(node->phase_state.y, node->phase_state.x);

            json << "    {\n";
            json << "      \"id\": " << node->id << ",\n";
            json << "      \"scale_level\": " << node->scale_level << ",\n";
            json << "      \"parent_id\": " << node->parent_id << ",\n";
            json << "      \"rotor_angle\": " << phase_angle << ",\n";
            json << "      \"rotor_cos\": " << node->phase_state.x << ",\n";
            json << "      \"rotor_sin\": " << node->phase_state.y << ",\n";
            json << "      \"coherence\": " << node->coherence << ",\n";
            json << "      \"is_fused\": " << (node->is_fused ? "true" : "false") << "\n";
            json << "    }" << (node_idx < node_map.size() - 1 ? "," : "") << "\n";
            node_idx++;
        }
        json << "  ],\n";

        json << "  \"edges\": [\n";
        std::vector<std::string> edge_strings;

        for (const auto& [id, node] : node_map) {
            for (int child_id : node->children_ids) {
                if (node_map.count(child_id)) {
                    const auto& child = node_map.at(child_id);
                    float p_angle = std::atan2(node->phase_state.y, node->phase_state.x);
                    float c_angle = std::atan2(child->phase_state.y, child->phase_state.x);
                    float delta_theta = c_angle - p_angle;

                    std::ostringstream edge_ss;
                    edge_ss << "    {\n"
                            << "      \"source\": " << node->id << ",\n"
                            << "      \"target\": " << child_id << ",\n"
                            << "      \"phase_shift\": " << delta_theta << "\n"
                            << "    }";
                    edge_strings.push_back(edge_ss.str());
                }
            }
        }

        for (size_t i = 0; i < edge_strings.size(); ++i) {
            json << edge_strings[i] << (i < edge_strings.size() - 1 ? "," : "") << "\n";
        }
        json << "  ]\n";
        json << "}\n";

        return json.str();
    }

private:
    static std::string phase_space_to_string(PhaseSpace space) {
        switch (space) {
            case PhaseSpace::Real_1D:     return "ℝ (Real 1D)";
            case PhaseSpace::Complex_2D:  return "ℂ (Complex 2D)";
            case PhaseSpace::Clifford_3D: return "Cℓ₃,₀ (Clifford 3D Spin)";
        }
        return "Unknown";
    }
};
