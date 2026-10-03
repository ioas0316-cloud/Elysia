#pragma once

#include <vector>
#include <cstddef>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace elysia::engine {

struct SensoryState {
    std::vector<float> k_audio;   // [Dim]
    float omega_audio{0.0f};
    float amplitude{0.0f};
    float imu_accel[3]{0.0f, 0.0f, 0.0f};
    std::vector<float> z_text;    // [Dim]
};

class TensorFieldResonator {
public:
    TensorFieldResonator(size_t batch_size, size_t num_points, size_t dim);
    ~TensorFieldResonator();

    TensorFieldResonator(const TensorFieldResonator&) = delete;
    TensorFieldResonator& operator=(const TensorFieldResonator&) = delete;
    TensorFieldResonator(TensorFieldResonator&&) noexcept;
    TensorFieldResonator& operator=(TensorFieldResonator&&) noexcept;

    void SetQueryPositions(const float* host_positions);
    void UpdateSensoryState(const SensoryState& state);
    void ComputeField(float current_time, float sigma = 0.15f);

    const float* GetHostDistances() const { return host_distances_.data(); }
    const float* GetHostNormals() const { return host_normals_.data(); }

    const float* GetDeviceDistances() const { return d_distances_; }
    const float* GetDeviceNormals() const { return d_normals_; }

private:
    size_t batch_size_;
    size_t num_points_;
    size_t dim_;

    std::vector<float> host_query_x_;
    SensoryState current_sensory_state_;

    std::vector<float> host_distances_;
    std::vector<float> host_normals_;

    float* d_query_x_{nullptr};
    float* d_k_audio_{nullptr};
    float* d_omega_audio_{nullptr};
    float* d_amplitude_{nullptr};
    float* d_imu_accel_{nullptr};
    float* d_z_text_{nullptr};

    float* d_distances_{nullptr};
    float* d_normals_{nullptr};

#ifdef WITH_CUDA
    cudaStream_t stream_{nullptr};
#endif

    void AllocateBuffers();
    void FreeBuffers();
    void ComputeCPUFallback(float current_time, float sigma);
};

} // namespace elysia::engine
