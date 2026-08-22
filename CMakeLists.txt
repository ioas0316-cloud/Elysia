cmake_minimum_required(VERSION 3.14)
project(causal_engine_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/causal_topology
)

# Core test binary
add_executable(test_preisach_tensor_field tests/cpp/test_preisach_tensor_field.cpp)
target_link_libraries(test_preisach_tensor_field PRIVATE OpenMP::OpenMP_CXX)

# pybind11 Python extension module
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (pybind11_DIR)
        find_package(pybind11 CONFIG HINTS ${pybind11_DIR})
    endif()
endif()

if (pybind11_FOUND)
    pybind11_add_module(causal_engine_py src/bindings/python_bindings.cpp)
    target_link_libraries(causal_engine_py PRIVATE OpenMP::OpenMP_CXX)
    set_target_properties(causal_engine_py PROPERTIES OUTPUT_NAME "causal_engine")
endif()
