#ifndef TOPIR_TYPES_HPP
#define TOPIR_TYPES_HPP

#include <string>
#include <vector>
#include <memory>
#include <sstream>

namespace elysia::topir {

enum class TypeKind {
    S3Field,         ///< S^3 4D Unit Sphere Manifold Field (Quaternion/Rotor)
    OrderField,      ///< Order Parameter Field \Phi bounded in [0, 1]
    TensorField,     ///< Tensor Field with rank R and dimension D
    VectorField3D,   ///< 3D Vector Field
    ScalarField      ///< Continuous Scalar Field
};

struct TopIRType {
    TypeKind kind;
    int rank = 0;       ///< Tensor rank R
    int dimension = 3;  ///< Spatial dimension D (default 3)
    std::string domain_name = "Grid3D";

    std::string to_string() const {
        switch (kind) {
            case TypeKind::S3Field:       return "S3Field";
            case TypeKind::OrderField:    return "OrderField<float>";
            case TypeKind::TensorField:   return "TensorField<R=" + std::to_string(rank) + ", D=" + std::to_string(dimension) + ">";
            case TypeKind::VectorField3D: return "VectorField3D";
            case TypeKind::ScalarField:   return "ScalarField";
        }
        return "UnknownType";
    }
};

struct TopIRNode;

/// Fiber Bundle Edge representing topological connectivity & continuous field flow between operator nodes
struct FiberBundleEdge {
    std::string edge_id;
    TopIRType type;
    std::string source_node_id;
    std::string target_node_id;
    int source_output_slot = 0;
    int target_input_slot = 0;
    bool preserves_topology = true;
    bool is_continuous_flow = true;
};

} // namespace elysia::topir

#endif // TOPIR_TYPES_HPP
