#include "../include/topir_runtime_engine.hpp"
#include "../isa_compiler/codegen/cpp_cuda_codegen.hpp"
#include <cmath>
#include <algorithm>

namespace elysia::runtime {

TopIRRuntimeEngine::TopIRRuntimeEngine(const RuntimeConfig& config)
    : config_(config) {
    initialize();
}

void TopIRRuntimeEngine::initialize() {
    size_t num_voxels = get_num_voxels();
    q_field_.assign(num_voxels * 4, 0.0f);
    v_field_.assign(num_voxels * 3, 0.0f);
    temp_field_.assign(num_voxels, config_.temp);

    // Initialize quaternions to identity rotor (x=0, y=0, z=0, w=1)
    for (size_t i = 0; i < num_voxels; ++i) {
        q_field_[i * 4 + 3] = 1.0f;
    }
}

void TopIRRuntimeEngine::step() {
    elysia::generated::step_system_cpu(
        q_field_.data(),
        v_field_.data(),
        temp_field_.data(),
        config_.grid_dim,
        config_.dt,
        config_.K_0
    );
}

} // namespace elysia::runtime

extern "C" {

ELYSIA_API void* elysia_topir_runtime_create(int grid_dim, float dt, float K_0) {
    elysia::runtime::RuntimeConfig config;
    config.grid_dim = grid_dim;
    config.dt = dt;
    config.K_0 = K_0;
    return new elysia::runtime::TopIRRuntimeEngine(config);
}

ELYSIA_API void elysia_topir_runtime_destroy(void* engine) {
    if (engine) {
        delete static_cast<elysia::runtime::TopIRRuntimeEngine*>(engine);
    }
}

ELYSIA_API void elysia_topir_runtime_step(void* engine) {
    if (engine) {
        static_cast<elysia::runtime::TopIRRuntimeEngine*>(engine)->step();
    }
}

ELYSIA_API float* elysia_topir_runtime_get_q_field(void* engine) {
    if (engine) {
        return static_cast<elysia::runtime::TopIRRuntimeEngine*>(engine)->get_q_field();
    }
    return nullptr;
}

ELYSIA_API float* elysia_topir_runtime_get_v_field(void* engine) {
    if (engine) {
        return static_cast<elysia::runtime::TopIRRuntimeEngine*>(engine)->get_v_field();
    }
    return nullptr;
}

}
