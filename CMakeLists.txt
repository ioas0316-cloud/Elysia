cmake_minimum_required(VERSION 3.14)
project(causal_engine_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/causal_topology
)

# Core C++ Causal Nexus Library
add_library(nexus_core STATIC
    src/nexus_betti.cpp
    src/3dgs_sh_eval.cpp
)
target_link_libraries(nexus_core PRIVATE OpenMP::OpenMP_CXX)

# Core test binaries
add_executable(test_preisach_tensor_field tests/cpp/test_preisach_tensor_field.cpp)
target_link_libraries(test_preisach_tensor_field PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_superconducting_kernel tests/cpp/test_superconducting_kernel.cpp)
target_link_libraries(test_superconducting_kernel PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_nexus_causal_engine tests/cpp/test_nexus_causal_engine.cpp)
target_link_libraries(test_nexus_causal_engine PRIVATE nexus_core OpenMP::OpenMP_CXX)

add_executable(nexus_shared_memory_bench src/nexus_shared_memory_bench.cpp)
target_link_libraries(nexus_shared_memory_bench PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_hyper_pixel_engine tests/cpp/test_hyper_pixel_engine.cpp)
target_link_libraries(test_hyper_pixel_engine PRIVATE nexus_core OpenMP::OpenMP_CXX)

add_executable(test_3dgs_sh_eval tests/cpp/test_3dgs_sh_eval.cpp)
target_link_libraries(test_3dgs_sh_eval PRIVATE nexus_core OpenMP::OpenMP_CXX)

add_executable(test_causal_field_accelerator tests/cpp/test_causal_field_accelerator.cpp)
target_link_libraries(test_causal_field_accelerator PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_closed_loop_governor tests/cpp/test_closed_loop_governor.cpp)
target_link_libraries(test_closed_loop_governor PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_meta_causal_map tests/cpp/test_meta_causal_map.cpp)
target_link_libraries(test_meta_causal_map PRIVATE OpenMP::OpenMP_CXX)

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
