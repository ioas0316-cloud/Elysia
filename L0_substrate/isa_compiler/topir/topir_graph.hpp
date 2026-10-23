#ifndef TOPIR_GRAPH_HPP
#define TOPIR_GRAPH_HPP

#include "topir_types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace elysia::topir {

enum class OperatorOpcode {
    DIFF_CURL,            ///< Spatial Differential operator: \nabla \times V
    DIFF_DIV,             ///< Spatial Divergence operator: \nabla \cdot T
    DIFF_GRAD,            ///< Spatial Gradient operator: \nabla \Phi
    PHASE_LOCK,           ///< Neighbor phase-locking torque operator: \Sigma_sync
    SCALE_COMPRESS,       ///< Scale logarithmic compression operator: \tau / (1 + \gamma \log(1 + ||\tau||^2))
    LANGEVIN_INTEGRATE,   ///< Stochastic Langevin rotator update on S^3
    LOCAL_COHERENCE,      ///< Order parameter \Phi coherence evaluation
    BLEND_BY_ORDER,       ///< Continuous Potential Field Branch Neutralization Blending
    NOISE_ROTATIONAL,     ///< Rotational Brownian noise generator
    CUSTOM_OPERATOR       ///< User-defined geometric operator
};

struct TopIRNode {
    std::string node_id;
    OperatorOpcode opcode;
    std::string op_name;
    TopIRType output_type;

    std::vector<std::string> input_edge_ids;
    std::vector<std::string> output_edge_ids;

    // Attributes
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, std::string> string_params;

    // Branch Neutralization specifics
    bool is_blended_branch = false;
    std::string blend_order_param_id;
    std::vector<std::string> blended_branches; // Branch case names (e.g. Solid, Liquid, Gas)

    std::string to_string() const {
        std::ostringstream ss;
        ss << "[" << node_id << "] " << op_name << " (" << output_type.to_string() << ")";
        if (is_blended_branch) {
            ss << " <Zero-Branch Blended by " << blend_order_param_id << ">";
        }
        return ss.str();
    }
};

class TopologicalGraphIR {
public:
    TopologicalGraphIR() = default;

    void add_node(const TopIRNode& node) {
        nodes_[node.node_id] = node;
        node_order_.push_back(node.node_id);
    }

    void add_edge(const FiberBundleEdge& edge) {
        edges_[edge.edge_id] = edge;
    }

    const TopIRNode* get_node(const std::string& node_id) const {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) return &it->second;
        return nullptr;
    }

    TopIRNode* get_node_mut(const std::string& node_id) {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) return &it->second;
        return nullptr;
    }

    const std::vector<std::string>& get_node_order() const {
        return node_order_;
    }

    const std::unordered_map<std::string, FiberBundleEdge>& get_edges() const {
        return edges_;
    }

    void remove_node(const std::string& node_id) {
        nodes_.erase(node_id);
        std::vector<std::string> new_order;
        for (const auto& id : node_order_) {
            if (id != node_id) new_order.push_back(id);
        }
        node_order_ = std::move(new_order);
    }

    std::string print_graph() const {
        std::ostringstream ss;
        ss << "================ Topological Graph IR (TopIR) ================\n";
        ss << "Nodes (" << nodes_.size() << "):\n";
        for (const auto& id : node_order_) {
            const auto& node = nodes_.at(id);
            ss << "  " << node.to_string() << "\n";
        }
        ss << "Fiber Bundle Edges (" << edges_.size() << "):\n";
        for (const auto& kv : edges_) {
            const auto& edge = kv.second;
            ss << "  Edge " << edge.edge_id << " [" << edge.type.to_string() << "]: "
               << edge.source_node_id << " -> " << edge.target_node_id << "\n";
        }
        ss << "=============================================================\n";
        return ss.str();
    }

private:
    std::unordered_map<std::string, TopIRNode> nodes_;
    std::unordered_map<std::string, FiberBundleEdge> edges_;
    std::vector<std::string> node_order_;
};

} // namespace elysia::topir

#endif // TOPIR_GRAPH_HPP
