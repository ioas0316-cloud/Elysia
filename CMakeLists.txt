cmake_minimum_required(VERSION 3.14)
project(causal_engine_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/causal_topology
    ${CMAKE_CURRENT_SOURCE_DIR}/emulation
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Core C++ Causal Nexus Library
add_library(nexus_core STATIC
    src/nexus_betti.cpp
    src/3dgs_sh_eval.cpp
)
target_link_libraries(nexus_core PRIVATE OpenMP::OpenMP_CXX)

# SIMD GPU ECS Library
add_library(simd_gpu_ecs STATIC
    src/simd_gpu_ecs/simd_torque_transmission_system.cpp
    src/simd_gpu_ecs/gpu_particle_frame_pipeline.cpp
    src/simd_gpu_ecs/indirect_dispatch_pipeline.cpp
)
target_include_directories(simd_gpu_ecs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(simd_gpu_ecs PRIVATE -mavx2 -mfma)
endif()

# Emulation Stage 2 Library
add_library(emulation_stage2 STATIC
    emulation/metric_field_engine.cpp
    emulation/metric_field_engine_flux.cpp
    emulation/atlas_manager.cpp
    emulation/spatiotemporal_pipeline.cpp
)

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

add_executable(test_arena_zero_copy tests/cpp/test_arena_zero_copy.cpp)
target_link_libraries(test_arena_zero_copy PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_spatiotemporal_pipeline tests/cpp/test_spatiotemporal_pipeline.cpp)
target_link_libraries(test_spatiotemporal_pipeline PRIVATE emulation_stage2 OpenMP::OpenMP_CXX)

add_executable(test_simd_gpu_ecs tests/test_simd_gpu_ecs.cpp)
target_link_libraries(test_simd_gpu_ecs PRIVATE simd_gpu_ecs)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_simd_gpu_ecs PRIVATE -mavx2 -mfma)
endif()

# CMake HLSL DXC Offline Compilation Automation Target
find_program(DXC_EXECUTABLE dxc PATHS "$ENV{WindowsSDKDir}/bin/$ENV{WindowsSDKLibVersion}/x64" "/usr/bin" "/usr/local/bin")

if(DXC_EXECUTABLE)
    file(GLOB HLSL_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/simd_gpu_ecs/*.hlsl")
    foreach(SHADER ${HLSL_SHADERS})
        get_filename_component(FILE_NAME ${SHADER} NAME_WE)
        add_custom_target(
            Compile_${FILE_NAME} ALL
            COMMAND ${DXC_EXECUTABLE} -E main -T cs_6_6 -Fo "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.cso" "${SHADER}"
            COMMENT "Compiling Shader: ${FILE_NAME}.hlsl"
        )
    endforeach()
endif()

# Elysia Dual Pillars Hardware Pipeline Library
set(ELYSIA_HARDWARE_CU_SOURCES
    src/phase_lock_evaluator.cu
    src/sensory_tensor_encoder.cu
    src/sensory_quantizer.cu
    src/codebook_online_trainer.cu
    src/multi_gpu_nvlink_router.cu
)

set_source_files_properties(${ELYSIA_HARDWARE_CU_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")

add_library(elysia_hardware STATIC
    src/d3d12_causal_page_allocator.cpp
    src/d3d12_cuda_interop.cpp
    src/d3d12_cuda_semaphore.cpp
    src/causal_memory_cache_manager.cpp
    src/elysian_engine_main_loop.cpp
    ${ELYSIA_HARDWARE_CU_SOURCES}
)
target_include_directories(elysia_hardware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_elysian_hardware_pipeline tests/cpp/test_elysian_hardware_pipeline.cpp)
target_link_libraries(test_elysian_hardware_pipeline PRIVATE elysia_hardware OpenMP::OpenMP_CXX)

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
