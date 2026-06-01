cmake_minimum_required(VERSION 3.15)
project(elysia_core_poc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(elysia_core
    main.cpp
    phase_transformer.cpp
    fractal_mirror.cpp
    stator_dynamics.cpp
    static_context_bridge.cpp
)
