#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <elysia/cl31_sta.hpp>

namespace py = pybind11;

py::tuple cl31_weave_pybind(
    py::array_t<float, py::array::c_style | py::array::forcecast> A_s,
    py::array_t<float, py::array::c_style | py::array::forcecast> A_v0,
    py::array_t<float, py::array::c_style | py::array::forcecast> A_v1,
    py::array_t<float, py::array::c_style | py::array::forcecast> A_v2,
    py::array_t<float, py::array::c_style | py::array::forcecast> A_v3,
    py::array_t<float, py::array::c_style | py::array::forcecast> B_s,
    py::array_t<float, py::array::c_style | py::array::forcecast> B_v0,
    py::array_t<float, py::array::c_style | py::array::forcecast> B_v1,
    py::array_t<float, py::array::c_style | py::array::forcecast> B_v2,
    py::array_t<float, py::array::c_style | py::array::forcecast> B_v3)
{
    auto r_s = A_s.unchecked<1>();
    size_t N = r_s.shape(0);

    auto Out_s  = py::array_t<float>(N);
    auto Out_b0 = py::array_t<float>(N);
    auto Out_b1 = py::array_t<float>(N);
    auto Out_b2 = py::array_t<float>(N);
    auto Out_b3 = py::array_t<float>(N);
    auto Out_b4 = py::array_t<float>(N);
    auto Out_b5 = py::array_t<float>(N);

    elysia::cl31_weave_soa_cpu(
        A_s.data(), A_v0.data(), A_v1.data(), A_v2.data(), A_v3.data(),
        B_s.data(), B_v0.data(), B_v1.data(), B_v2.data(), B_v3.data(),
        Out_s.mutable_data(), Out_b0.mutable_data(), Out_b1.mutable_data(), Out_b2.mutable_data(),
        Out_b3.mutable_data(), Out_b4.mutable_data(), Out_b5.mutable_data(),
        N);

    return py::make_tuple(Out_s, Out_b0, Out_b1, Out_b2, Out_b3, Out_b4, Out_b5);
}

PYBIND11_MODULE(elysia_cl31_pybind, m) {
    m.doc() = "Elysia Cl(3,1) STA CPU / PyBind Engine Module";
    m.def("cl31_weave", &cl31_weave_pybind, "Cl(3,1) Geometric Product Weave Engine (CPU/Numpy)");
}
