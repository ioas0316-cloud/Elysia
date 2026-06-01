cmake_minimum_required(VERSION 3.15)
project(ElysiaCorePoC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(core)

add_executable(elysia_core core/resonance_daemon.cpp) # Just linking it as the core executable for now as PoC
