cmake_minimum_required(VERSION 3.18)
project(causal_engine_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA compiler
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61 75 86)
    endif()
endif()

option(BUILD_D3D12_RENDERER "Build Direct3D12 Volume Raymarcher GUI Engine (Windows only)" OFF)

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 QUIET)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/isa_compiler
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/causal_topology
    ${CMAKE_CURRENT_SOURCE_DIR}/emulation
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Core C++ Causal Nexus Library
add_library(nexus_core STATIC
    src/nexus_betti.cpp
    src/3dgs_sh_eval.cpp
    src/topir_runtime_engine.cpp
)
target_link_libraries(nexus_core PRIVATE OpenMP::OpenMP_CXX)

# TopIR Runtime Engine Shared Library
add_library(topir_runtime SHARED
    src/topir_runtime_engine.cpp
)
target_link_libraries(topir_runtime PRIVATE OpenMP::OpenMP_CXX)

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

# TopIR C++ Test Binary
add_executable(test_topir_cpp_pipeline tests/cpp/test_topir_cpp_pipeline.cpp)
target_link_libraries(test_topir_cpp_pipeline PRIVATE nexus_core OpenMP::OpenMP_CXX)

# CMake HLSL DXC Offline Compilation Automation Target
find_program(DXC_EXECUTABLE dxc PATHS "$ENV{WindowsSDKDir}/bin/$ENV{WindowsSDKLibVersion}/x64" "/usr/bin" "/usr/local/bin")

if(DXC_EXECUTABLE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ElysiaFusedKernel.hlsl")
        add_custom_target(
            Compile_ElysiaFusedKernel ALL
            COMMAND ${DXC_EXECUTABLE} -E CS_EvolveSystem -T cs_6_5 -Fo "${CMAKE_CURRENT_BINARY_DIR}/ElysiaFusedKernel.cso" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ElysiaFusedKernel.hlsl"
            COMMENT "Compiling Shader: ElysiaFusedKernel.hlsl"
        )
    endif()
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

add_executable(test_elysia_relaxation_solver tests/cpp/test_elysia_relaxation_solver.cpp)
target_link_libraries(test_elysia_relaxation_solver PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_holographic_engine tests/cpp/test_holographic_engine.cpp)
target_include_directories(test_holographic_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_holographic_engine PRIVATE OpenMP::OpenMP_CXX)

# Elysia FOC & Clifford Wave Computing targets (Host Fallback / C++)
set_source_files_properties(src/elysia_foc_kernel.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")
set_source_files_properties(src/elysia_autonomic_controller.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")

add_executable(test_elysia_foc_pipeline
    tests/cpp/test_elysia_foc_pipeline.cpp
    src/elysia_foc_kernel.cu
    src/elysia_autonomic_controller.cu
    src/elysia_scheduler.cpp
)
target_include_directories(test_elysia_foc_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_elysia_foc_pipeline PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_elysia_autonomic_consolidation
    tests/cpp/test_elysia_autonomic_consolidation.cpp
    src/elysia_autonomic_controller.cu
    src/elysia_foc_kernel.cu
    src/elysia_scheduler.cpp
)
target_include_directories(test_elysia_autonomic_consolidation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_elysia_autonomic_consolidation PRIVATE OpenMP::OpenMP_CXX)

add_executable(elysia_foc_main
    src/elysia_foc_main.cpp
    src/elysia_foc_kernel.cu
    src/elysia_autonomic_controller.cu
    src/elysia_scheduler.cpp
)
target_include_directories(elysia_foc_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(elysia_foc_main PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# Elysia Core Benchmarks & CUDA Targets
# -----------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_executable(elysia_core_bench src/elysia_core_loop.cu)
    target_compile_options(elysia_core_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
            -Xcompiler -O3,-Wall
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -Wall
        >
    )
    if(TARGET Eigen3::Eigen)
        target_link_libraries(elysia_core_bench PRIVATE Eigen3::Eigen)
    endif()

    add_executable(elysia_pinned_async_stream src/elysia_pinned_async_stream.cu)
    target_compile_options(elysia_pinned_async_stream PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
            -Xcompiler -O3,-Wall
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -Wall
        >
    )
    if(TARGET Eigen3::Eigen)
        target_link_libraries(elysia_pinned_async_stream PRIVATE Eigen3::Eigen)
    endif()

    add_executable(elysia_integrated_bench src/elysia_integrated_bench.cu)
    target_compile_options(elysia_integrated_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
            -Xcompiler -O3,-Wall
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -Wall
        >
    )

    add_executable(elysia_unified_hologram_engine src/elysia_unified_hologram_engine.cu)
    target_include_directories(elysia_unified_hologram_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(elysia_unified_hologram_engine PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
            -Xcompiler -O3,-Wall
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -O3
            -Wall
        >
    )
else()
    # Host fallback executables compiled as C++ with host stub
    set_source_files_properties(src/elysia_core_loop.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")
    add_executable(elysia_core_bench src/elysia_core_loop.cu)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(elysia_core_bench PRIVATE Eigen3::Eigen)
    endif()

    set_source_files_properties(src/elysia_pinned_async_stream.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")
    add_executable(elysia_pinned_async_stream src/elysia_pinned_async_stream.cu)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(elysia_pinned_async_stream PRIVATE Eigen3::Eigen)
    endif()

    set_source_files_properties(src/elysia_integrated_bench.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")
    add_executable(elysia_integrated_bench src/elysia_integrated_bench.cu)

    set_source_files_properties(src/elysia_unified_hologram_engine.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -include elysia/cuda_host_stub.h")
    add_executable(elysia_unified_hologram_engine src/elysia_unified_hologram_engine.cu)
    target_include_directories(elysia_unified_hologram_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Optional Direct3D12 Renderer Target
if (WIN32 AND BUILD_D3D12_RENDERER)
    add_executable(ElysiaVolumeRaymarcher
        src/elysia_d3d12_raymarcher.cpp
    )
    target_link_libraries(ElysiaVolumeRaymarcher PRIVATE d3d12.lib dxgi.lib d3dcompiler.lib nexus_core)
endif()

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
