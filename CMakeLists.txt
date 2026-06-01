cmake_minimum_required(VERSION 3.15)
project(ElysiaCorePoC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(core)

add_executable(elysia_core core/resonance_daemon.cpp core/elysia_bridge.cpp)
target_link_libraries(elysia_core rt) # For shm_open
