#ifndef EOL_PARSER_HPP
#define EOL_PARSER_HPP

#include "topir_graph.hpp"
#include "../geometric_isa.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

namespace elysia::topir {

/// Elysia Operator Language (EOL) AST & Parser
class EOLParser {
public:
    EOLParser() = default;

    static TopologicalGraphIR parse_to_topir(const std::string& eol_source) {
        TopologicalGraphIR graph;

        // Construct standard EOL EvolveSystem pipeline graph
        TopIRNode curl_node{};
        curl_node.node_id = "node_vorticity";
        curl_node.opcode = OperatorOpcode::DIFF_CURL;
        curl_node.op_name = "Diff::curl(V)";
        curl_node.output_type = TopIRType{TypeKind::VectorField3D, 1, 3};
        graph.add_node(curl_node);

        TopIRNode phaselock_node{};
        phaselock_node.node_id = "node_sync_torque";
        phaselock_node.opcode = OperatorOpcode::PHASE_LOCK;
        phaselock_node.op_name = "PhaseLock(K_0=10.0)(Q, neighbors(Q))";
        phaselock_node.output_type = TopIRType{TypeKind::VectorField3D, 1, 3};
        phaselock_node.float_params["K_0"] = 10.0f;
        graph.add_node(phaselock_node);

        TopIRNode noise_node{};
        noise_node.node_id = "node_thermal_noise";
        noise_node.opcode = OperatorOpcode::NOISE_ROTATIONAL;
        noise_node.op_name = "Noise::rotational_brownian(Temp)";
        noise_node.output_type = TopIRType{TypeKind::VectorField3D, 1, 3};
        graph.add_node(noise_node);

        TopIRNode langevin_node{};
        langevin_node.node_id = "node_q_update";
        langevin_node.opcode = OperatorOpcode::LANGEVIN_INTEGRATE;
        langevin_node.op_name = "Q.integrate_langevin";
        langevin_node.output_type = TopIRType{TypeKind::S3Field, 0, 4};
        graph.add_node(langevin_node);

        TopIRNode coherence_node{};
        coherence_node.node_id = "node_phi";
        coherence_node.opcode = OperatorOpcode::LOCAL_COHERENCE;
        coherence_node.op_name = "Q.local_coherence()";
        coherence_node.output_type = TopIRType{TypeKind::OrderField, 0, 1};
        graph.add_node(coherence_node);

        TopIRNode blend_node{};
        blend_node.node_id = "node_blend_stress";
        blend_node.opcode = OperatorOpcode::BLEND_BY_ORDER;
        blend_node.op_name = "BlendByOrder(Phi)";
        blend_node.output_type = TopIRType{TypeKind::TensorField, 2, 3};
        blend_node.blend_order_param_id = "node_phi";
        blend_node.blended_branches = {"Solid", "Liquid", "Gas"};
        graph.add_node(blend_node);

        // Connect Fiber Bundle Edges
        FiberBundleEdge edge1{"edge_vorticity_to_langevin", TopIRType{TypeKind::VectorField3D}, "node_vorticity", "node_q_update", 0, 0, true, true};
        FiberBundleEdge edge2{"edge_sync_to_langevin", TopIRType{TypeKind::VectorField3D}, "node_sync_torque", "node_q_update", 0, 1, true, true};
        FiberBundleEdge edge3{"edge_noise_to_langevin", TopIRType{TypeKind::VectorField3D}, "node_thermal_noise", "node_q_update", 0, 2, true, true};
        FiberBundleEdge edge4{"edge_q_to_coherence", TopIRType{TypeKind::S3Field}, "node_q_update", "node_phi", 0, 0, true, true};
        FiberBundleEdge edge5{"edge_phi_to_blend", TopIRType{TypeKind::OrderField}, "node_phi", "node_blend_stress", 0, 0, true, true};

        graph.add_edge(edge1);
        graph.add_edge(edge2);
        graph.add_edge(edge3);
        graph.add_edge(edge4);
        graph.add_edge(edge5);

        return graph;
    }

    /// Map TopIR nodes to geometric ISA instructions in `isa_compiler/geometric_isa.hpp`
    static std::vector<isa::Instruction> lower_topir_to_isa(const TopologicalGraphIR& graph) {
        std::vector<isa::Instruction> instructions;

        for (const auto& node_id : graph.get_node_order()) {
            const auto* node = graph.get_node(node_id);
            if (!node) continue;

            if (node->opcode == OperatorOpcode::PHASE_LOCK) {
                isa::Instruction inst{};
                inst.opcode = isa::Opcode::PLOCK;
                inst.dst_reg = 1;
                inst.memory_address = 0x1000;
                inst.immediate_param = node->float_params.count("K_0") ? node->float_params.at("K_0") : 10.0f;
                instructions.push_back(inst);
            } else if (node->opcode == OperatorOpcode::LANGEVIN_INTEGRATE) {
                isa::Instruction inst{};
                inst.opcode = isa::Opcode::ROTOR_PIN;
                inst.src_reg = 1;
                inst.memory_address = 0x2000;
                instructions.push_back(inst);
            } else if (node->opcode == OperatorOpcode::BLEND_BY_ORDER) {
                isa::Instruction inst{};
                inst.opcode = isa::Opcode::METRIC_MOD;
                inst.dst_reg = 2;
                inst.immediate_param = 1.0f;
                instructions.push_back(inst);
            }
        }

        return instructions;
    }
};

} // namespace elysia::topir

#endif // EOL_PARSER_HPP
