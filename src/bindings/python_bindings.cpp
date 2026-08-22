#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "causal_engine/core/types.hpp"
#include "causal_engine/core/preisach_soa.hpp"
#include "causal_engine/extraction/attractor_layer.hpp"
#include "causal_engine/reasoning/backtracer.hpp"
#include "causal_engine/feedback/closed_loop.hpp"
#include "causal_engine/feedback/causal_impedance.hpp"

namespace py = pybind11;
using namespace causal_engine;

PYBIND11_MODULE(causal_engine, m) {
    m.doc() = "C++ High-Performance Bi-directional Causal Engine Python Binding";

    // 1. Core Structs Binding
    py::class_<MacroSymbolNode>(m, "MacroSymbolNode")
        .def(py::init<>())
        .def_readwrite("node_id", &MacroSymbolNode::node_id)
        .def_readwrite("pivot_alpha", &MacroSymbolNode::pivot_alpha)
        .def_readwrite("pivot_beta", &MacroSymbolNode::pivot_beta)
        .def_readwrite("axiom_rigidity", &MacroSymbolNode::axiom_rigidity)
        .def_readwrite("current_state_sr", &MacroSymbolNode::current_state_sr);

    py::class_<CausalEdge>(m, "CausalEdge")
        .def(py::init<>())
        .def_readwrite("source_node_id", &CausalEdge::source_node_id)
        .def_readwrite("target_node_id", &CausalEdge::target_node_id)
        .def_readwrite("causal_weight", &CausalEdge::causal_weight)
        .def_readwrite("reluctance", &CausalEdge::reluctance);

    // 2. PreisachTensorFieldSoA Binding (NumPy / PyTorch Zero-Copy Interop)
    py::class_<PreisachTensorFieldSoA>(m, "PreisachTensorFieldSoA")
        .def(py::init<size_t, size_t>(), py::arg("num_nodes") = 64, py::arg("hysterons_per_dim") = 8)
        .def_readwrite("num_nodes", &PreisachTensorFieldSoA::num_nodes)
        .def_readwrite("num_hysterons", &PreisachTensorFieldSoA::num_hysterons)
        .def_readwrite("density_weights", &PreisachTensorFieldSoA::density_weights)
        .def_readwrite("alpha_grid", &PreisachTensorFieldSoA::alpha_grid)
        .def_readwrite("beta_grid", &PreisachTensorFieldSoA::beta_grid)
        // Zero-Copy / Fast Injection of input signals
        .def("set_input_signals_from_numpy", [](PreisachTensorFieldSoA& self, py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
            py::buffer_info buf = input_array.request();
            if (static_cast<size_t>(buf.size) != self.num_nodes) {
                throw std::runtime_error("Input array size does not match num_nodes!");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            std::copy(ptr, ptr + self.num_nodes, self.input_signals.begin());
        })
        // Zero-Copy NumPy View of Remanence States (S_r)
        .def("get_remanence_as_numpy", [](PreisachTensorFieldSoA& self) {
            return py::array_t<float>(
                { self.num_nodes },
                { sizeof(float) },
                self.remanence_states.data(),
                py::cast(self) // keep_alive reference to ensure lifetime safety
            );
        });

    // GIL-Free OpenMP Preisach Field Update
    m.def("update_preisach_field", [](PreisachTensorFieldSoA& field) {
        py::gil_scoped_release release;
        UpdatePreisachTensorField(field);
    }, "Execute OpenMP/SIMD update on Preisach Tensor Field");

    // 3. AttractorExtractionLayer Binding
    py::class_<AttractorExtractionLayer>(m, "AttractorExtractionLayer")
        .def(py::init<>())
        .def("extract_causal_graph", [](AttractorExtractionLayer& self, const PreisachTensorFieldSoA& field, float threshold) {
            std::vector<MacroSymbolNode> nodes;
            std::vector<CausalEdge> edges;
            self.ExtractCausalGraph(field, nodes, edges, threshold);
            return py::make_tuple(nodes, edges);
        }, py::arg("field"), py::arg("threshold") = 0.4f);

    // 4. Enhanced CausalBacktracer Binding
    py::class_<EnhancedCausalBacktracer>(m, "CausalBacktracer")
        .def(py::init<>())
        .def("trace_minimal_impedance_path", &EnhancedCausalBacktracer::TraceMinimalImpedancePath,
             py::arg("goal_node_id"), py::arg("start_node_id"), py::arg("nodes"), py::arg("edges"))
        .def("trace_minimal_impedance_path_with_latency", &EnhancedCausalBacktracer::TraceMinimalImpedancePathWithLatency,
             py::arg("goal_node_id"), py::arg("start_node_id"), py::arg("nodes"), py::arg("edges"),
             py::arg("gamma_curvature") = 0.2f, py::arg("latency_damping") = 0.1f);

    // 5. ClosedLoopCausalEngine Binding
    py::class_<ClosedLoopCausalEngine>(m, "ClosedLoopCausalEngine")
        .def(py::init<>())
        .def("execute_and_adapt", [](ClosedLoopCausalEngine& self, const std::vector<uint32_t>& trajectory, const std::vector<MacroSymbolNode>& nodes, PreisachTensorFieldSoA& field, float threshold) {
            py::gil_scoped_release release;
            return self.ExecuteAndAdaptTrajectory(trajectory, nodes, field, threshold);
        }, py::arg("trajectory"), py::arg("nodes"), py::arg("field"), py::arg("threshold") = 0.2f);

    // 6. Impedance Evaluation & Meta-Constraint Binding
    py::class_<ImpedanceResult>(m, "ImpedanceResult")
        .def(py::init<>())
        .def_readwrite("trajectory_curvature", &ImpedanceResult::trajectory_curvature)
        .def_readwrite("topological_phase_diff", &ImpedanceResult::topological_phase_diff)
        .def_readwrite("latency_damped_friction", &ImpedanceResult::latency_damped_friction)
        .def_readwrite("resonance_score", &ImpedanceResult::resonance_score)
        .def_readwrite("requires_rule_mutation", &ImpedanceResult::requires_rule_mutation);

    py::class_<CausalImpedanceEvaluator>(m, "CausalImpedanceEvaluator")
        .def_static("compute_curvature", &CausalImpedanceEvaluator::ComputeTrajectoryCurvature)
        .def_static("compute_phase_diff", &CausalImpedanceEvaluator::ComputeTopologicalPhaseDiscrepancy)
        .def_static("evaluate_impedance", &CausalImpedanceEvaluator::EvaluateImpedance,
                    py::arg("nodes"), py::arg("candidate_trajectory"), py::arg("target_trajectory"),
                    py::arg("gamma_curvature") = 0.3f, py::arg("latency_damping") = 0.2f, py::arg("friction_threshold") = 0.45f);

    py::class_<MetaConstraintRule>(m, "MetaConstraintRule")
        .def(py::init<>())
        .def_readwrite("max_reluctance_threshold", &MetaConstraintRule::max_reluctance_threshold)
        .def_readwrite("min_rigidity_threshold", &MetaConstraintRule::min_rigidity_threshold)
        .def_readwrite("alpha_boundary_min", &MetaConstraintRule::alpha_boundary_min)
        .def_readwrite("alpha_boundary_max", &MetaConstraintRule::alpha_boundary_max)
        .def_readwrite("beta_boundary_min", &MetaConstraintRule::beta_boundary_min)
        .def_readwrite("beta_boundary_max", &MetaConstraintRule::beta_boundary_max)
        .def_readwrite("curvature_penalty_weight", &MetaConstraintRule::curvature_penalty_weight);

    py::class_<MetaConstraintMutator>(m, "MetaConstraintMutator")
        .def(py::init<>())
        .def("get_current_rule", &MetaConstraintMutator::GetCurrentRule)
        .def("get_mutation_count", &MetaConstraintMutator::GetMutationCount)
        .def("mutate_rule", &MetaConstraintMutator::MutateRule)
        .def("filter_nodes", &MetaConstraintMutator::FilterNodes)
        .def("filter_edges", &MetaConstraintMutator::FilterEdges);
}
