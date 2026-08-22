#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "causal_engine/core/types.hpp"
#include "causal_engine/core/preisach_soa.hpp"
#include "causal_engine/extraction/attractor_layer.hpp"
#include "causal_engine/reasoning/backtracer.hpp"
#include "causal_engine/feedback/closed_loop.hpp"

namespace py = pybind11;

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
}
