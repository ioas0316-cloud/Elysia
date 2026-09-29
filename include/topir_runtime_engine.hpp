#ifndef ELYSIA_TOPIR_RUNTIME_HPP
#define ELYSIA_TOPIR_RUNTIME_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>

#if defined(_WIN32)
  #define ELYSIA_API __declspec(dllexport)
#else
  #define ELYSIA_API __attribute__((visibility("default")))
#endif

namespace elysia::runtime {

struct RuntimeConfig {
    int grid_dim = 64; // Grid size: grid_dim x grid_dim x grid_dim
    float dt = 0.005f;
    float K_0 = 10.0f;
    float temp = 0.1f;
};

class TopIRRuntimeEngine {
public:
    TopIRRuntimeEngine(const RuntimeConfig& config = RuntimeConfig());
    ~TopIRRuntimeEngine() = default;

    void initialize();
    void step();

    float* get_q_field() { return q_field_.data(); }
    float* get_v_field() { return v_field_.data(); }
    float* get_temp_field() { return temp_field_.data(); }

    const float* get_q_field() const { return q_field_.data(); }
    const float* get_v_field() const { return v_field_.data(); }
    const float* get_temp_field() const { return temp_field_.data(); }

    size_t get_num_voxels() const { return static_cast<size_t>(config_.grid_dim) * config_.grid_dim * config_.grid_dim; }

private:
    RuntimeConfig config_;
    std::vector<float> q_field_;    // 4 components per voxel: x, y, z, w
    std::vector<float> v_field_;    // 3 components per voxel: vx, vy, vz
    std::vector<float> temp_field_; // 1 component per voxel
};

} // namespace elysia::runtime

extern "C" {

ELYSIA_API void* elysia_topir_runtime_create(int grid_dim, float dt, float K_0);
ELYSIA_API void elysia_topir_runtime_destroy(void* engine);
ELYSIA_API void elysia_topir_runtime_step(void* engine);
ELYSIA_API float* elysia_topir_runtime_get_q_field(void* engine);
ELYSIA_API float* elysia_topir_runtime_get_v_field(void* engine);

}

#endif // ELYSIA_TOPIR_RUNTIME_HPP
