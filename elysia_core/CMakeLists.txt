cmake_minimum_required(VERSION 3.15)
project(ElysiaCore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(ElysiaCore
    src/main.cpp
    src/EnvironmentGenerator.cpp
    src/VariableResistanceKnob.cpp
    src/FractalMirrorRoom.cpp
    src/CoreKernel.cpp
)
