#pragma once

#include <iostream>
#include <iomanip>
#include <algorithm>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

class VramAdaptiveController {
private:
    float current_free_ratio_{1.0f};      // 현재 남은 VRAM 비율 [0.0, 1.0]
    float current_prune_threshold_{0.1f}; // 계산된 동적 임계값

    // [입력 변수] 남은 VRAM 비율 영역
    const float R_CRITICAL_FREE = 0.10f; // 10% 미만 남았을 때: 최대로 공격적인 가지치기
    const float R_SAFE_FREE     = 0.50f; // 50% 이상 남았을 때: 최소한의 정착(Gentle) 가지치기

    // [출력 변수] noise_prune_threshold 한계값
    const float T_GENTLE       = 0.05f; // 자원 풍부 시: 미세 노이즈만 제거 (해상도 유지)
    const float T_AGGRESSIVE   = 0.45f; // 자원 고갈 시: 강한 노이즈 및 약한 위상까지 제거 (압축률 극대화)

public:
    VramAdaptiveController() {
        update_vram_status();
    }

    void update_vram_status() {
        size_t free_mem = 0, total_mem = 0;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

        if (err == cudaSuccess && total_mem > 0) {
            current_free_ratio_ = static_cast<float>(free_mem) / static_cast<float>(total_mem);
        } else {
            current_free_ratio_ = 0.3f;
        }

        compute_adaptive_threshold();
    }

    void set_simulated_free_ratio(float ratio) {
        current_free_ratio_ = std::clamp(ratio, 0.0f, 1.0f);
        compute_adaptive_threshold();
    }

private:
    void compute_adaptive_threshold() {
        float clamped_ratio = std::clamp(current_free_ratio_, R_CRITICAL_FREE, R_SAFE_FREE);
        float normalized_danger = (R_SAFE_FREE - clamped_ratio) / (R_SAFE_FREE - R_CRITICAL_FREE);
        current_prune_threshold_ = T_GENTLE + normalized_danger * (T_AGGRESSIVE - T_GENTLE);
    }

public:
    [[nodiscard]] float get_dynamic_threshold() const { return current_prune_threshold_; }
    [[nodiscard]] float get_free_ratio() const { return current_free_ratio_; }

    void print_status() const {
        std::cout << "[VramProprioception] Free VRAM: " << std::fixed << std::setprecision(1)
                  << (current_free_ratio_ * 100.0f) << "% -> "
                  << "Adaptive Prune Threshold: " << std::setprecision(3)
                  << current_prune_threshold_ << " (Mode: "
                  << (current_prune_threshold_ > (T_GENTLE + T_AGGRESSIVE) * 0.5f ? "Aggressive" : "Gentle")
                  << ")\n";
    }
};
