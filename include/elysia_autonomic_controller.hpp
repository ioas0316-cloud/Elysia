#ifndef ELYSIA_AUTONOMIC_CONTROLLER_HPP
#define ELYSIA_AUTONOMIC_CONTROLLER_HPP

#include <iostream>
#include <algorithm>
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

enum class AutonomicMode {
    Sympathetic,     // Sympathetic mode: Melting, active noise exploration, error tracking
    Parasympathetic  // Parasympathetic mode: Crystallization, sleep/memory consolidation, energy conservation
};

struct EngineMetrics {
    float mismatch_error;   // Top-down vs Bottom-up wave mismatch (0.0 ~ 1.0)
    float coherence;        // Clifford rotor phase coherence C (0.0 ~ 1.0)
    float perceived_stress; // GPU load & VRAM bottleneck stress (0.0 ~ 1.0)
};

class AutonomicStateController {
private:
    AutonomicMode current_mode_{AutonomicMode::Parasympathetic};
    float acetylcholine_lvl_{0.1f};
    float dopamine_rpe_lvl_{0.5f};

    const float mismatch_high_th_{0.65f};
    const float mismatch_low_th_{0.15f};
    const float coherence_min_th_{0.80f};

public:
    AutonomicStateController() = default;

    void update(const EngineMetrics& metrics) {
        if (current_mode_ == AutonomicMode::Parasympathetic) {
            if (metrics.mismatch_error > mismatch_high_th_) {
                current_mode_ = AutonomicMode::Sympathetic;
                acetylcholine_lvl_ = 1.0f;
            }
        } else {
            if (metrics.mismatch_error < mismatch_low_th_ && metrics.coherence >= coherence_min_th_) {
                current_mode_ = AutonomicMode::Parasympathetic;
                acetylcholine_lvl_ = 0.02f;
            }
        }

        if (metrics.perceived_stress > 0.9f && current_mode_ == AutonomicMode::Sympathetic) {
            acetylcholine_lvl_ *= 0.5f;
        }
    }

    [[nodiscard]] AutonomicMode get_mode() const { return current_mode_; }
    [[nodiscard]] float get_ach_level() const { return acetylcholine_lvl_; }
    [[nodiscard]] bool is_parasympathetic() const { return current_mode_ == AutonomicMode::Parasympathetic; }
};

class VramAdaptiveController {
private:
    float current_free_ratio_{1.0f};
    float current_prune_threshold_{0.1f};

    const float R_CRITICAL_FREE = 0.10f;
    const float R_SAFE_FREE     = 0.50f;

    const float T_GENTLE       = 0.05f;
    const float T_AGGRESSIVE   = 0.45f;

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

    void set_simulated_vram_free_ratio(float ratio) {
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
    float get_dynamic_threshold() const { return current_prune_threshold_; }
    float get_free_ratio() const { return current_free_ratio_; }
};

#ifdef __cplusplus
extern "C" {
#endif

void launch_parasympathetic_consolidation_kernel(
    float2* d_rotors,
    float* d_coherence_map,
    float ach_level,
    float noise_prune_threshold,
    int num_rotors,
    cudaStream_t stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_AUTONOMIC_CONTROLLER_HPP
