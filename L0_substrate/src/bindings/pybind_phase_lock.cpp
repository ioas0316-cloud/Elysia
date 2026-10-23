#include <torch/extension.h>
#include "phase_lock_engine.hpp"
#include "sensory_phase_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Elysia Internal Metric Field Phase-Lock Engine C++/CUDA Extension";

    // Bind float4 and uint4 helper types
    py::class_<Elysia::float4>(m, "Float4")
        .def(py::init<>())
        .def(py::init([](float x, float y, float z, float w) {
            return Elysia::float4{x, y, z, w};
        }), py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f, py::arg("w") = 0.0f)
        .def(py::init([](const std::vector<float>& v) {
            float x = v.size() > 0 ? v[0] : 0.0f;
            float y = v.size() > 1 ? v[1] : 0.0f;
            float z = v.size() > 2 ? v[2] : 0.0f;
            float w = v.size() > 3 ? v[3] : 0.0f;
            return Elysia::float4{x, y, z, w};
        }))
        .def_readwrite("x", &Elysia::float4::x)
        .def_readwrite("y", &Elysia::float4::y)
        .def_readwrite("z", &Elysia::float4::z)
        .def_readwrite("w", &Elysia::float4::w)
        .def("__getitem__", [](const Elysia::float4& self, size_t idx) {
            if (idx == 0) return self.x;
            if (idx == 1) return self.y;
            if (idx == 2) return self.z;
            if (idx == 3) return self.w;
            throw py::index_error("Float4 index out of range");
        })
        .def("__len__", [](const Elysia::float4&) { return 4; });

    py::implicitly_convertible<std::vector<float>, Elysia::float4>();

    py::class_<Elysia::uint4>(m, "Uint4")
        .def(py::init<>())
        .def(py::init([](uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
            return Elysia::uint4{x, y, z, w};
        }), py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0, py::arg("w") = 0)
        .def(py::init([](const std::vector<uint32_t>& v) {
            uint32_t x = v.size() > 0 ? v[0] : 0;
            uint32_t y = v.size() > 1 ? v[1] : 0;
            uint32_t z = v.size() > 2 ? v[2] : 0;
            uint32_t w = v.size() > 3 ? v[3] : 0;
            return Elysia::uint4{x, y, z, w};
        }))
        .def_readwrite("x", &Elysia::uint4::x)
        .def_readwrite("y", &Elysia::uint4::y)
        .def_readwrite("z", &Elysia::uint4::z)
        .def_readwrite("w", &Elysia::uint4::w)
        .def("__getitem__", [](const Elysia::uint4& self, size_t idx) {
            if (idx == 0) return self.x;
            if (idx == 1) return self.y;
            if (idx == 2) return self.z;
            if (idx == 3) return self.w;
            throw py::index_error("Uint4 index out of range");
        })
        .def("__len__", [](const Elysia::uint4&) { return 4; });

    py::implicitly_convertible<std::vector<uint32_t>, Elysia::uint4>();

    // Original Phase-Lock Engine Bindings
    py::class_<ElysiaEngine::PhaseLockEnginePipeline::Config>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("R_cut", &ElysiaEngine::PhaseLockEnginePipeline::Config::R_cut)
        .def_readwrite("dt", &ElysiaEngine::PhaseLockEnginePipeline::Config::dt)
        .def_readwrite("phi_solid", &ElysiaEngine::PhaseLockEnginePipeline::Config::phi_solid)
        .def_readwrite("phi_gas", &ElysiaEngine::PhaseLockEnginePipeline::Config::phi_gas)
        .def_readwrite("gamma_m", &ElysiaEngine::PhaseLockEnginePipeline::Config::gamma_m)
        .def_readwrite("alpha", &ElysiaEngine::PhaseLockEnginePipeline::Config::alpha)
        .def_readwrite("beta", &ElysiaEngine::PhaseLockEnginePipeline::Config::beta)
        .def_readwrite("tau_c", &ElysiaEngine::PhaseLockEnginePipeline::Config::tau_c)
        .def_readwrite("c0", &ElysiaEngine::PhaseLockEnginePipeline::Config::c0)
        .def_readwrite("lambda_c", &ElysiaEngine::PhaseLockEnginePipeline::Config::lambda_c);

    py::class_<ElysiaEngine::PhaseLockEnginePipeline>(m, "PhaseLockEnginePipeline")
        .def(py::init<const ElysiaEngine::PhaseLockEnginePipeline::Config&>(), py::arg("config") = ElysiaEngine::PhaseLockEnginePipeline::Config())
        .def("step", &ElysiaEngine::PhaseLockEnginePipeline::step,
             py::arg("X"), py::arg("V"), py::arg("row_ptr"), py::arg("col_idx"),
             py::arg("C_edge"), py::arg("M_edge"))
        .def_property_readonly("config", &ElysiaEngine::PhaseLockEnginePipeline::config);

    // Pure Sensory-Phase Core Bindings
    py::class_<Elysia::SensoryPhaseConfig>(m, "SensoryPhaseConfig")
        .def(py::init<>())
        .def_readwrite("lambda_1", &Elysia::SensoryPhaseConfig::lambda_1)
        .def_readwrite("lambda_2", &Elysia::SensoryPhaseConfig::lambda_2)
        .def_readwrite("lambda_3", &Elysia::SensoryPhaseConfig::lambda_3)
        .def_readwrite("kappa_1", &Elysia::SensoryPhaseConfig::kappa_1)
        .def_readwrite("alpha", &Elysia::SensoryPhaseConfig::alpha)
        .def_readwrite("beta", &Elysia::SensoryPhaseConfig::beta)
        .def_readwrite("gamma_max", &Elysia::SensoryPhaseConfig::gamma_max)
        .def_readwrite("phi_liquid", &Elysia::SensoryPhaseConfig::phi_liquid)
        .def_readwrite("phi_solid", &Elysia::SensoryPhaseConfig::phi_solid)
        .def_readwrite("tau_shear", &Elysia::SensoryPhaseConfig::tau_shear);

    py::class_<Elysia::SensoryStreamInput>(m, "SensoryStreamInput")
        .def(py::init<>())
        .def_readwrite("position_tension", &Elysia::SensoryStreamInput::position_tension)
        .def_readwrite("velocity_dtension", &Elysia::SensoryStreamInput::velocity_dtension)
        .def_readwrite("audio_spectrum", &Elysia::SensoryStreamInput::audio_spectrum)
        .def_readwrite("acceleration_grad", &Elysia::SensoryStreamInput::acceleration_grad);

    py::class_<Elysia::SpacetimeNodeBoundary>(m, "SpacetimeNodeBoundary")
        .def(py::init<>())
        .def_readwrite("metric_diag", &Elysia::SpacetimeNodeBoundary::metric_diag)
        .def_readwrite("metric_offdiag", &Elysia::SpacetimeNodeBoundary::metric_offdiag)
        .def_readwrite("csr_topology", &Elysia::SpacetimeNodeBoundary::csr_topology);

    py::class_<Elysia::TensorFieldDiagnostics>(m, "TensorFieldDiagnostics")
        .def(py::init<>())
        .def_readwrite("position_gdi", &Elysia::TensorFieldDiagnostics::position_gdi)
        .def_readwrite("saddle_hessian", &Elysia::TensorFieldDiagnostics::saddle_hessian)
        .def_readwrite("classification_st", &Elysia::TensorFieldDiagnostics::classification_st);

    py::class_<Elysia::SensoryPhaseCorePipeline>(m, "SensoryPhaseCorePipeline")
        .def(py::init<const Elysia::SensoryPhaseConfig&>(), py::arg("config") = Elysia::SensoryPhaseConfig())
        .def("process_frame_cpu", [](Elysia::SensoryPhaseCorePipeline& self, const std::vector<Elysia::SensoryStreamInput>& inputs) {
            std::vector<Elysia::SpacetimeNodeBoundary> nodes;
            std::vector<Elysia::TensorFieldDiagnostics> diagnostics;
            self.process_frame_cpu(inputs, nodes, diagnostics);
            return std::make_pair(nodes, diagnostics);
        })
        .def_property_readonly("config", &Elysia::SensoryPhaseCorePipeline::get_config);
}
