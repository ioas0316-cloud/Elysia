#ifndef TOPIR_PASSES_HPP
#define TOPIR_PASSES_HPP

#include "topir_graph.hpp"
#include <iostream>
#include <vector>

namespace elysia::topir {

/// Pass 1: Algebraic Isomorphism Reduction Pass
/// Collapses consecutive operations (e.g. Diff::curl and PhaseLock) into a single Fused Isomorphic Kernel Node
class AlgebraicIsomorphismReductionPass {
public:
    static bool run(TopologicalGraphIR& graph) {
        bool modified = false;
        std::vector<std::string> order = graph.get_node_order();

        for (size_t i = 0; i + 1 < order.size(); ++i) {
            auto node_a = graph.get_node(order[i]);
            auto node_b = graph.get_node(order[i+1]);

            if (!node_a || !node_b) continue;

            // Check if Diff::curl followed by PhaseLock can be algebraically fused
            if (node_a->opcode == OperatorOpcode::DIFF_CURL && node_b->opcode == OperatorOpcode::PHASE_LOCK) {
                // Perform categorical operation fusion: F(g o f)
                std::string fused_id = "Fused_" + node_a->node_id + "_" + node_b->node_id;

                TopIRNode fused_node{};
                fused_node.node_id = fused_id;
                fused_node.opcode = OperatorOpcode::PHASE_LOCK; // Fused Operator
                fused_node.op_name = "Fused_Vorticity_PhaseLock";
                fused_node.output_type = node_b->output_type;
                fused_node.float_params = node_b->float_params;
                fused_node.float_params["fused_curl_alpha"] = 1.0f;

                // Transfer input edges
                fused_node.input_edge_ids = node_a->input_edge_ids;

                graph.add_node(fused_node);
                modified = true;
                break; // One fusion pass iteration
            }
        }

        return modified;
    }
};

/// Pass 2: Continuous Branch Neutralization Pass
/// Converts discrete if/else comparison logic into continuous potential field weighted blending (\Phi blending)
class ContinuousBranchNeutralizationPass {
public:
    static bool run(TopologicalGraphIR& graph) {
        bool modified = false;
        std::vector<std::string> order = graph.get_node_order();

        for (const auto& id : order) {
            auto node = graph.get_node_mut(id);
            if (!node) continue;

            if (node->opcode == OperatorOpcode::BLEND_BY_ORDER) {
                // Eliminate discrete branches by ensuring weights are continuous sigmoids/saturate functions
                node->is_blended_branch = true;
                node->string_params["blend_mode"] = "Continuous_Sigmoid_Weight_Field";
                node->float_params["solid_threshold"] = 0.7f;
                node->float_params["gas_threshold"] = 0.3f;
                node->float_params["steepness"] = 10.0f;
                modified = true;
            }
        }

        return modified;
    }
};

} // namespace elysia::topir

#endif // TOPIR_PASSES_HPP
