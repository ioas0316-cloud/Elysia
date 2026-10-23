#include "elysia/core/ElysiaTensorFieldResonator.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef WITH_CUDA
extern "C" void launch_multisensory_sdf_kernel_impl(
    const float* query_x, const float* k_audio, const float* omega_audio,
    const float* amp_audio, const float* imu_accel, const float* z_text,
    float* d_out, int batch_size, int n_points, int dim, float t_curr, float sigma,
    cudaStream_t stream
);
#endif

namespace elysia::engine {

TensorFieldResonator::TensorFieldResonator(size_t batch_size, size_t num_points, size_t dim)
    : batch_size_(batch_size), num_points_(num_points), dim_(dim) {
    host_distances_.resize(batch_size_ * num_points_);
    host_normals_.resize(batch_size_ * num_points_ * 3);
    host_query_x_.resize(batch_size_ * num_points_ * dim_);

    current_sensory_state_.k_audio.resize(dim_, 0.0f);
    current_sensory_state_.z_text.resize(dim_, 0.0f);

    AllocateBuffers();
}

TensorFieldResonator::~TensorFieldResonator() {
    FreeBuffers();
}

void TensorFieldResonator::AllocateBuffers() {
#ifdef WITH_CUDA
    cudaStreamCreate(&stream_);
    size_t query_bytes = batch_size_ * num_points_ * dim_ * sizeof(float);
    size_t vector_bytes = batch_size_ * dim_ * sizeof(float);
    size_t scalar_bytes = batch_size_ * sizeof(float);
    size_t imu_bytes = batch_size_ * 3 * sizeof(float);
    size_t out_bytes = batch_size_ * num_points_ * sizeof(float);
    size_t normal_bytes = batch_size_ * num_points_ * 3 * sizeof(float);

    cudaMalloc(&d_query_x_, query_bytes);
    cudaMalloc(&d_k_audio_, vector_bytes);
    cudaMalloc(&d_omega_audio_, scalar_bytes);
    cudaMalloc(&d_amplitude_, scalar_bytes);
    cudaMalloc(&d_imu_accel_, imu_bytes);
    cudaMalloc(&d_z_text_, vector_bytes);
    cudaMalloc(&d_distances_, out_bytes);
    cudaMalloc(&d_normals_, normal_bytes);
#endif
}

void TensorFieldResonator::FreeBuffers() {
#ifdef WITH_CUDA
    if (d_query_x_) cudaFree(d_query_x_);
    if (d_k_audio_) cudaFree(d_k_audio_);
    if (d_omega_audio_) cudaFree(d_omega_audio_);
    if (d_amplitude_) cudaFree(d_amplitude_);
    if (d_imu_accel_) cudaFree(d_imu_accel_);
    if (d_z_text_) cudaFree(d_z_text_);
    if (d_distances_) cudaFree(d_distances_);
    if (d_normals_) cudaFree(d_normals_);
    if (stream_) cudaStreamDestroy(stream_);
#endif
}

void TensorFieldResonator::SetQueryPositions(const float* host_positions) {
    size_t total_elements = batch_size_ * num_points_ * dim_;
    std::memcpy(host_query_x_.data(), host_positions, total_elements * sizeof(float));
#ifdef WITH_CUDA
    if (d_query_x_) {
        cudaMemcpyAsync(d_query_x_, host_positions, total_elements * sizeof(float), cudaMemcpyHostToDevice, stream_);
    }
#endif
}

void TensorFieldResonator::UpdateSensoryState(const SensoryState& state) {
    current_sensory_state_ = state;
#ifdef WITH_CUDA
    if (d_k_audio_) {
        cudaMemcpyAsync(d_k_audio_, state.k_audio.data(), dim_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_omega_audio_, &state.omega_audio, sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_amplitude_, &state.amplitude, sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_imu_accel_, state.imu_accel, 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_z_text_, state.z_text.data(), dim_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    }
#endif
}

void TensorFieldResonator::ComputeField(float current_time, float sigma) {
#ifdef WITH_CUDA
    if (d_query_x_) {
        launch_multisensory_sdf_kernel_impl(
            d_query_x_, d_k_audio_, d_omega_audio_, d_amplitude_,
            d_imu_accel_, d_z_text_, d_distances_,
            static_cast<int>(batch_size_), static_cast<int>(num_points_), static_cast<int>(dim_),
            current_time, sigma, stream_
        );
        cudaMemcpyAsync(host_distances_.data(), d_distances_, batch_size_ * num_points_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return;
    }
#endif
    ComputeCPUFallback(current_time, sigma);
}

void TensorFieldResonator::ComputeCPUFallback(float current_time, float sigma) {
    for (size_t b = 0; b < batch_size_; ++b) {
        float ax = current_sensory_state_.imu_accel[0];
        float ay = current_sensory_state_.imu_accel[1];
        float az = current_sensory_state_.imu_accel[2];

        float omega_t = std::fmod(current_sensory_state_.omega_audio * current_time, 6.28318530718f);

        for (size_t p = 0; p < num_points_; ++p) {
            size_t idx = (b * num_points_ + p) * dim_;
            size_t out_idx = b * num_points_ + p;

            float x0 = host_query_x_[idx + 0] + ax * 0.1f;
            float x1 = host_query_x_[idx + 1] + ay * 0.1f;
            float x2 = host_query_x_[idx + 2] + az * 0.1f;

            float norm_sq = x0 * x0 + x1 * x1 + x2 * x2;
            float k_dot_x = x0 * current_sensory_state_.k_audio[0] +
                            x1 * current_sensory_state_.k_audio[1] +
                            x2 * current_sensory_state_.k_audio[2];
            float z_dot_x = x0 * current_sensory_state_.z_text[0] +
                            x1 * current_sensory_state_.z_text[1] +
                            x2 * current_sensory_state_.z_text[2];

            for (size_t d = 3; d < dim_; ++d) {
                float x_val = host_query_x_[idx + d];
                norm_sq += x_val * x_val;
                if (d < current_sensory_state_.k_audio.size()) k_dot_x += x_val * current_sensory_state_.k_audio[d];
                if (d < current_sensory_state_.z_text.size()) z_dot_x += x_val * current_sensory_state_.z_text[d];
            }

            float norm_x = std::sqrt(norm_sq + 1e-8f);
            float d_base = norm_x - 1.0f;

            float total_phase = k_dot_x + z_dot_x - omega_t;
            float mask = std::exp(-std::abs(d_base) / sigma);
            float perturbation = current_sensory_state_.amplitude * mask * std::cos(total_phase);

            host_distances_[out_idx] = d_base + perturbation;

            // Analytic Normal approximation
            if (p * 3 + 2 < host_normals_.size()) {
                host_normals_[out_idx * 3 + 0] = x0 / norm_x;
                host_normals_[out_idx * 3 + 1] = x1 / norm_x;
                host_normals_[out_idx * 3 + 2] = x2 / norm_x;
            }
        }
    }
}

} // namespace elysia::engine
