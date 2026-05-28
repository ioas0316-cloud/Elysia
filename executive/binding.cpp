// [행정부] pybind11 연결 관로
// C++ Native Binding을 통해 GPU VRAM 면의 포인터 주소를 파이썬에서 직접 참조

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// CUDA 함수 선언
extern void run_cuda_kernel(float* input, float* output, int size);

// Python 바인딩 함수
void launch_membrane_calculation(py::array_t<float> input_array, py::array_t<float> output_array) {
    py::buffer_info buf_in = input_array.request();
    py::buffer_info buf_out = output_array.request();

    float* ptr_in = static_cast<float*>(buf_in.ptr);
    float* ptr_out = static_cast<float*>(buf_out.ptr);

    int size = buf_in.size;

    // TODO: 호출 CUDA 래퍼 연결
    // run_cuda_kernel(ptr_in, ptr_out, size);
    std::cout << "[Executive] VRAM 직접 참조 및 연산 요청 (Binding) - Size: " << size << std::endl;
}

PYBIND11_MODULE(elysia_core, m) {
    m.doc() = "Elysia Core PyBind11 Plugin";
    m.def("launch_membrane", &launch_membrane_calculation, "A function that runs CUDA kernel directly");
}
