#ifndef ELYSIA_SCHEDULER_HPP
#define ELYSIA_SCHEDULER_HPP

#include <cmath>
#include <iostream>
#include <algorithm>
#include "vram_monitor.hpp"
#include "elysia_foc_kernel.h"

// Dynamic GABA Threshold Regulator (Closed-Loop PID Control)
class DynamicGABARegulator {
private:
    float omega_th_;       // Current GABA Threshold (default 0.35f)
    float integral_error_;
    float prev_error_;

    float kp_;
    float ki_;
    float kd_;
    float target_load_;    // Target cognitive load (e.g. 0.40f)

public:
    DynamicGABARegulator(float initial_th = 0.35f, float target_load = 0.40f,
                         float kp = 0.15f, float ki = 0.02f, float kd = 0.05f)
        : omega_th_(initial_th), integral_error_(0.0f), prev_error_(0.0f),
          kp_(kp), ki_(ki), kd_(kd), target_load_(target_load) {}

    float update_gaba_threshold(const CognitiveLoadMetrics& metrics, float dt = 0.01f) {
        float current_load = 0.6f * metrics.phase_entropy + 0.4f * metrics.active_node_ratio;
        float error = current_load - target_load_;

        integral_error_ += error * dt;
        float derivative_error = (dt > 0.0f) ? ((error - prev_error_) / dt) : 0.0f;
        prev_error_ = error;

        float delta_omega = kp_ * error + ki_ * integral_error_ + kd_ * derivative_error;
        omega_th_ = std::clamp(omega_th_ + delta_omega, 0.10f, 0.80f);
        return omega_th_;
    }

    float get_current_threshold() const { return omega_th_; }
};

// Acetylcholine (ACh) Wave Controller for Mismatch & Neuromodulation
class AChWaveController {
private:
    float ach_level_;       // ACh concentration [0.0, 1.0]
    float gamma_decay_;     // ACh decay rate
    float alpha_gain_;      // Mismatch spike gain
    float eta_base_;        // Base learning rate

public:
    AChWaveController(float decay = 0.05f, float gain = 0.8f, float base_lr = 0.01f)
        : ach_level_(0.0f), gamma_decay_(decay), alpha_gain_(gain), eta_base_(base_lr) {}

    void update_state(float error_spike, float dt = 0.01f) {
        float dACh = (-gamma_decay_ * ach_level_ + alpha_gain_ * error_spike) * dt;
        ach_level_ = std::clamp(ach_level_ + dACh, 0.0f, 1.0f);
    }

    void get_signal_weights(float& w_top, float& w_bot) const {
        w_top = 1.0f - 0.75f * ach_level_;
        w_bot = 1.0f + 1.25f * ach_level_;
    }

    float get_dynamic_learning_rate() const {
        return eta_base_ * (1.0f + 5.0f * ach_level_ * ach_level_);
    }

    float get_phase_diffusion_coefficient() const {
        return std::exp(2.5f * ach_level_);
    }

    float get_ach_level() const { return ach_level_; }
    void set_ach_level(float val) { ach_level_ = std::clamp(val, 0.0f, 1.0f); }
};

// FOC Real-time Pipeline Scheduler
class ElysiaFOCScheduler {
private:
    float vram_limit_mb_;
    float safety_margin_ratio_;
    float threshold_mb_;
    float lambda_gain_;
    cudaStream_t stream_;
    DynamicGABARegulator gaba_regulator_;
    AChWaveController ach_controller_;

public:
    ElysiaFOCScheduler(float vram_limit_mb = 3072.0f, float safety_margin = 0.15f, float lambda_gain = 0.005f)
        : vram_limit_mb_(vram_limit_mb),
          safety_margin_ratio_(safety_margin),
          lambda_gain_(lambda_gain),
          stream_(nullptr),
          gaba_regulator_(0.35f, 0.40f),
          ach_controller_(0.05f, 0.8f, 0.01f)
    {
        threshold_mb_ = vram_limit_mb_ * (1.0f - safety_margin_ratio_);
        cudaStreamCreate(&stream_);
    }

    ~ElysiaFOCScheduler() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // FOC Flux Weakening factor calculation (gamma_d)
    float compute_flux_weakening_factor(const VRAMState& vram) const {
        if (vram.used_mb <= threshold_mb_) {
            return 1.0f; // VRAM safe: 100% core context preserved
        }
        float v_err = vram.used_mb - threshold_mb_;
        float gamma_d = std::exp(-lambda_gain_ * v_err);
        return std::max(gamma_d, 0.15f); // Minimum 15% flux floor
    }

    // Asynchronous step execution
    void step(const float* d_abc, const float* d_angles, float* d_dq0, int batch_size,
              float simulated_vram_mb = -1.0f) {
        VRAMState vram = (simulated_vram_mb >= 0.0f)
            ? VRAMTracker::get_simulated_state(simulated_vram_mb, vram_limit_mb_)
            : VRAMTracker::get_current_state();

        float gamma_d = compute_flux_weakening_factor(vram);
        launch_elysia_foc_kernel(d_abc, d_angles, d_dq0, gamma_d, batch_size, stream_);
    }

    void synchronize() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
    }

    cudaStream_t get_stream() const { return stream_; }
    DynamicGABARegulator& get_gaba_regulator() { return gaba_regulator_; }
    AChWaveController& get_ach_controller() { return ach_controller_; }
    float get_vram_limit_mb() const { return vram_limit_mb_; }
    float get_threshold_mb() const { return threshold_mb_; }
};

#endif // ELYSIA_SCHEDULER_HPP
