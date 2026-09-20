#include <torch/extension.h>
#include "phase_lock_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Elysia Internal Metric Field Phase-Lock Engine C++/CUDA Extension";

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
}
