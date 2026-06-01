cmake_minimum_required(VERSION 3.15)
project(ElysiaCorePoC CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and SIMD flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Enable AVX-512 if supported by compiler (Optional but recommended)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512vl -mavx512dq")
endif()

add_executable(elysia_core main.cpp phase_transformer.cpp fractal_mirror.cpp stator_dynamics.cpp)
