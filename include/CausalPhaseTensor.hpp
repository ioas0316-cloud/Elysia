#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

// CUDA Kernel declaration
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void k_singularity_rotor_bypass(
    float2* __restrict__ rotors,
    float*  __restrict__ curvature_map,
    const float bypass_angle,
    const int num_rotors
);
#else
inline void k_singularity_rotor_bypass_host(
    float2* rotors,
    float* curvature_map,
    float bypass_angle,
    int num_rotors
) {
    for (int i = 0; i < num_rotors; ++i) {
        float2 r = rotors[i];
        bool is_singular = std::isnan(r.x) || std::isnan(r.y) || std::isinf(r.x) || std::isinf(r.y);
        if (is_singular) {
            float half_angle = bypass_angle * 0.5f;
            float cos_half = std::cos(half_angle);
            float sin_half = std::sin(half_angle);
            float2 base_rotor = make_float2(1.0f, 0.0f);

            float new_x = base_rotor.x * cos_half - base_rotor.y * sin_half;
            float new_y = base_rotor.x * sin_half + base_rotor.y * cos_half;

            rotors[i] = make_float2(new_x, new_y);
            curvature_map[i] = 1.0f;
        } else {
            curvature_map[i] = 0.0f;
        }
    }
}
#endif

enum class PhaseSpace {
    Real_1D,     // 실수 축: 선형 기울기 및 스칼라 인과 (A -> B)
    Complex_2D,  // 복소 평면: 위상 회전 및 순환적 인과 (A <-> B)
    Clifford_3D  // 클리포드 대수 Cℓ₃,₀: 공간 스핀 및 입체적 인과 구조
};

class CausalPhaseTensor {
private:
    PhaseSpace current_space_{PhaseSpace::Real_1D};
    size_t num_elements_{0};

    // CUDA 디바이스 메모리 버퍼
    float2* d_rotors_{nullptr};
    float*  d_curvature_map_{nullptr};

    // 상전이 제어 매개변수
    float gradient_stagnation_{0.0f}; // 기울기 정체율 (0.0 ~ 1.0)
    float mismatch_error_{0.0f};      // 불일치 오차

public:
    CausalPhaseTensor(size_t num_elements) : num_elements_(num_elements) {
        cudaMalloc(reinterpret_cast<void**>(&d_rotors_), num_elements_ * sizeof(float2));
        cudaMalloc(reinterpret_cast<void**>(&d_curvature_map_), num_elements_ * sizeof(float));

        std::vector<float2> init_rotors(num_elements_, make_float2(1.0f, 0.0f));
        std::vector<float> init_curvatures(num_elements_, 0.0f);

        cudaMemcpy(d_rotors_, init_rotors.data(), num_elements_ * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curvature_map_, init_curvatures.data(), num_elements_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~CausalPhaseTensor() {
        if (d_rotors_) cudaFree(d_rotors_);
        if (d_curvature_map_) cudaFree(d_curvature_map_);
    }

    void evaluate_phase_transition(float new_gradient_stagnation, float new_mismatch_error) {
        gradient_stagnation_ = new_gradient_stagnation;
        mismatch_error_ = new_mismatch_error;

        switch (current_space_) {
            case PhaseSpace::Real_1D:
                if (gradient_stagnation_ > 0.80f || mismatch_error_ > 0.50f) {
                    current_space_ = PhaseSpace::Complex_2D;
                    std::cout << "[CausalPhaseTensor] Phase Transition: ℝ (Real 1D) -> ℂ (Complex 2D Rotation)\n";
                }
                break;

            case PhaseSpace::Complex_2D:
                if (mismatch_error_ > 0.85f) {
                    current_space_ = PhaseSpace::Clifford_3D;
                    std::cout << "[CausalPhaseTensor] Phase Transition: ℂ (Complex 2D) -> Cℓ₃,₀ (Clifford 3D Spin)\n";
                } else if (mismatch_error_ < 0.10f && gradient_stagnation_ < 0.20f) {
                    current_space_ = PhaseSpace::Real_1D;
                    std::cout << "[CausalPhaseTensor] Phase Relaxation: ℂ (Complex 2D) -> ℝ (Real 1D)\n";
                }
                break;

            case PhaseSpace::Clifford_3D:
                if (mismatch_error_ < 0.40f) {
                    current_space_ = PhaseSpace::Complex_2D;
                    std::cout << "[CausalPhaseTensor] Phase Relaxation: Cℓ₃,₀ (Clifford 3D) -> ℂ (Complex 2D)\n";
                }
                break;
        }
    }

    void resolve_singularities(cudaStream_t stream, float bypass_angle = 1.5707963f) {
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        int block_size = 256;
        int grid_size = static_cast<int>((num_elements_ + block_size - 1) / block_size);
        k_singularity_rotor_bypass<<<grid_size, block_size, 0, stream>>>(
            d_rotors_, d_curvature_map_, bypass_angle, static_cast<int>(num_elements_)
        );
#else
        k_singularity_rotor_bypass_host(d_rotors_, d_curvature_map_, bypass_angle, static_cast<int>(num_elements_));
#endif
    }

    void inject_singularity_for_test(size_t index) {
        if (index < num_elements_) {
            float2 nan_val = make_float2(NAN, NAN);
            cudaMemcpy(d_rotors_ + index, &nan_val, sizeof(float2), cudaMemcpyHostToDevice);
        }
    }

    [[nodiscard]] PhaseSpace get_current_space() const { return current_space_; }
    [[nodiscard]] float2* get_rotors_device_ptr() { return d_rotors_; }
    [[nodiscard]] float* get_curvature_device_ptr() { return d_curvature_map_; }
    [[nodiscard]] size_t get_num_elements() const { return num_elements_; }
};
