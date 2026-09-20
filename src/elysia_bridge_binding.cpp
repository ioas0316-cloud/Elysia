/*
 * elysia_bridge_binding.cpp
 * C++/CUDA PyBind11 Module for Elysia Phase-Locking & Dynamic Metric Field
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>

#ifdef HAS_CUDA
#include <cuda_runtime.h>

struct CausalNode {
    int id;
    float phase_state;
    float energy;
};

__global__ void phase_lock_kernel(CausalNode* nodes, float* metric_matrix, int num_nodes, int target_id, float ext_phase, float coupling_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float phase_diff = ext_phase - nodes[target_id].phase_state;

    if (idx == target_id) {
        nodes[idx].phase_state += coupling_k * sinf(phase_diff);
        nodes[idx].phase_state = fmodf(nodes[idx].phase_state + 2.0f * M_PI, 2.0f * M_PI);
    } else {
        int matrix_idx = target_id * num_nodes + idx;
        float dist = metric_matrix[matrix_idx];
        float influence = expf(-dist);
        nodes[idx].phase_state += coupling_k * influence * sinf(nodes[target_id].phase_state - nodes[idx].phase_state);
        nodes[idx].phase_state = fmodf(nodes[idx].phase_state + 2.0f * M_PI, 2.0f * M_PI);
    }
}
#endif

namespace py = pybind11;

class CppElysiaBridge {
private:
    int num_nodes;
    std::vector<float> phases;
    std::vector<float> metric_matrix;

public:
    CppElysiaBridge(int n) : num_nodes(n), phases(n, 0.0f), metric_matrix(n * n, 1.0f) {
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < num_nodes; ++j) {
                metric_matrix[i * num_nodes + j] = std::abs(i - j) * 0.2f + 0.1f;
            }
        }
    }

    void inject_external_phase(int target_id, float ext_phase, float coupling_k = 0.2f) {
        if (target_id < 0 || target_id >= num_nodes) return;

        float phase_diff = ext_phase - phases[target_id];
        phases[target_id] += coupling_k * std::sin(phase_diff);
        phases[target_id] = std::fmod(phases[target_id] + 2.0f * M_PI, 2.0f * M_PI);

        for (int j = 0; j < num_nodes; ++j) {
            if (j != target_id) {
                float dist = metric_matrix[target_id * num_nodes + j];
                float influence = std::exp(-dist);
                phases[j] += coupling_k * influence * std::sin(phases[target_id] - phases[j]);
                phases[j] = std::fmod(phases[j] + 2.0f * M_PI, 2.0f * M_PI);
            }
        }
    }

    py::array_t<float> get_phases() {
        auto result = py::array_t<float>(num_nodes);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        for (int i = 0; i < num_nodes; ++i) {
            ptr[i] = phases[i];
        }
        return result;
    }

    void set_metric_matrix(py::array_t<float> input_matrix) {
        py::buffer_info buf = input_matrix.request();
        if (buf.size == num_nodes * num_nodes) {
            float* ptr = static_cast<float*>(buf.ptr);
            std::copy(ptr, ptr + num_nodes * num_nodes, metric_matrix.begin());
        }
    }
};

PYBIND11_MODULE(elysia_cpp_bridge, m) {
    m.doc() = "Elysia C++ Native Bridge";
    py::class_<CppElysiaBridge>(m, "CppElysiaBridge")
        .def(py::init<int>(), py::arg("num_nodes") = 256)
        .def("inject_external_phase", &CppElysiaBridge::inject_external_phase, py::arg("target_id"), py::arg("ext_phase"), py::arg("coupling_k") = 0.2f)
        .def("get_phases", &CppElysiaBridge::get_phases)
        .def("set_metric_matrix", &CppElysiaBridge::set_metric_matrix);
}
