#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "causal_rotor_tensor.hpp"

namespace py = pybind11;
using namespace elysia;

PYBIND11_MODULE(elysia_causal_rotor_cuda, m) {
    m.doc() = "Elysia Causal Rotor Tensor & Phase-Lock Dynamical System Engine";

    py::class_<Float4>(m, "Float4")
        .def(py::init<float, float, float, float>(), py::arg("x")=0.0f, py::arg("y")=0.0f, py::arg("z")=0.0f, py::arg("w")=1.0f)
        .def_readwrite("x", &Float4::x)
        .def_readwrite("y", &Float4::y)
        .def_readwrite("z", &Float4::z)
        .def_readwrite("w", &Float4::w);

    py::class_<Float3>(m, "Float3")
        .def(py::init<float, float, float>(), py::arg("x")=0.0f, py::arg("y")=0.0f, py::arg("z")=0.0f)
        .def_readwrite("x", &Float3::x)
        .def_readwrite("y", &Float3::y)
        .def_readwrite("z", &Float3::z);

    py::class_<RotorNodeData>(m, "RotorNodeData")
        .def(py::init<>())
        .def_readwrite("quaternion", &RotorNodeData::quaternion)
        .def_readwrite("angular_velocity", &RotorNodeData::angularVelocity)
        .def_readwrite("damping_beta", &RotorNodeData::dampingBeta)
        .def_readwrite("spatial_hash_key", &RotorNodeData::spatialHashKey)
        .def_readwrite("usage_frequency", &RotorNodeData::usageFrequency)
        .def_readwrite("is_attractor", &RotorNodeData::isAttractor)
        .def_readwrite("vram_slab_address", &RotorNodeData::vramSlabAddress);

    py::class_<Morton3D>(m, "Morton3D")
        .def_static("encode", &Morton3D::encode, py::arg("x"), py::arg("y"), py::arg("z"))
        .def_static("decode", [](uint64_t code) {
            uint32_t x = 0, y = 0, z = 0;
            Morton3D::decode(code, x, y, z);
            return py::make_tuple(x, y, z);
        }, py::arg("code"));

    py::class_<CausalRotorTensorSystem>(m, "CausalRotorTensorSystem")
        .def(py::init<uint32_t>(), py::arg("max_nodes") = 65536)
        .def("add_rotor_node", &CausalRotorTensorSystem::addRotorNode,
             py::arg("grid_x"), py::arg("grid_y"), py::arg("grid_z"), py::arg("damping_beta") = 0.05f)
        .def("connect_nodes", &CausalRotorTensorSystem::connectNodes,
             py::arg("source_idx"), py::arg("target_idx"), py::arg("gear_ratio"))
        .def("inject_impulse", &CausalRotorTensorSystem::injectImpulse,
             py::arg("node_idx"), py::arg("wx"), py::arg("wy"), py::arg("wz"))
        .def("step_simulation", &CausalRotorTensorSystem::stepSimulation,
             py::arg("delta_time"), py::arg("lock_tolerance") = 0.01f)
        .def("get_attractor_count", &CausalRotorTensorSystem::getAttractorCount)
        .def("get_node_count", &CausalRotorTensorSystem::getNodeCount)
        .def("get_node_buffer", &CausalRotorTensorSystem::getNodeBuffer);
}
