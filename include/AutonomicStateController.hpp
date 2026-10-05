#pragma once

#include <iostream>
#include <algorithm>

enum class AutonomicMode {
    Sympathetic,     // 교감 신경: 노이즈 탐색, 로터 융해(Melting), 오차 추적
    Parasympathetic  // 부교감 신경: 결정화(Crystallization), 수면/메모리 정화, 에너지 절감
};

struct EngineMetrics {
    float mismatch_error{0.0f};   // Top-down vs Bottom-up 파동 불일치 (0.0 ~ 1.0)
    float coherence{0.5f};        // 클리포드 로터 위상 결맞음 지표 C (0.0 ~ 1.0)
    float perceived_stress{0.0f}; // GPU 연산 가부하 및 메모리 병목 스트레스
};

class AutonomicStateController {
private:
    AutonomicMode current_mode_{AutonomicMode::Parasympathetic};

    // 생화학 모듈레이터 대응 매개변수
    float acetylcholine_lvl_{0.1f};  // ACh: 높아지면 탐색/융해(교감), 낮아지면 결정화(부교감)
    float dopamine_rpe_lvl_{0.5f};    // RPE: 보상 예측 오차 기반 결맞음 강화 파라미터

    // 임계값 (Thresholds)
    const float mismatch_high_th_{0.65f};  // 교감 모드 전환 (오차 폭발)
    const float mismatch_low_th_{0.15f};   // 부교감 모드 전환 (오차 수렴)
    const float coherence_min_th_{0.80f};   // 부교감 결정화 전환을 위한 최소 결맞음

public:
    AutonomicStateController() = default;

    void update(const EngineMetrics& metrics) {
        if (current_mode_ == AutonomicMode::Parasympathetic) {
            // [부교감 -> 교감] 오차가 급증하거나 예측이 틀렸을 때 긴급 탐색 모드 진입
            if (metrics.mismatch_error > mismatch_high_th_) {
                current_mode_ = AutonomicMode::Sympathetic;
                acetylcholine_lvl_ = 1.0f; // ACh 급증 -> 로터 결맞음 약화 (Melting 시작)
                std::cout << "[AutonomicState] Transition -> SYMPATHETIC (Melting Mode Activated)\n";
            }
        } else {
            // [교감 -> 부교감] 오차가 낮아지고 위상 결맞음이 유의미하게 높아지면 결정화 모드 진입
            if (metrics.mismatch_error < mismatch_low_th_ && metrics.coherence >= coherence_min_th_) {
                current_mode_ = AutonomicMode::Parasympathetic;
                acetylcholine_lvl_ = 0.02f; // ACh Quenching -> 강한 위상 고정 (Crystallization 시작)
                std::cout << "[AutonomicState] Transition -> PARASYMPATHETIC (Consolidation Mode Activated)\n";
            }
        }

        // 스트레스 상한선에 따른 감쇠 제어
        if (metrics.perceived_stress > 0.9f && current_mode_ == AutonomicMode::Sympathetic) {
            // 자가 스트레스 과다 시 강제 부교감 쿨다운 유도
            acetylcholine_lvl_ *= 0.5f; // 탐색 감도 강제 저하
        }
    }

    void inject_metacognitive_stress(float stress_level) {
        if (stress_level > 0.6f) {
            if (current_mode_ == AutonomicMode::Parasympathetic) {
                current_mode_ = AutonomicMode::Sympathetic;
                acetylcholine_lvl_ = std::min(1.0f, acetylcholine_lvl_ + stress_level * 0.5f);
            }
        } else {
            acetylcholine_lvl_ = std::max(0.02f, acetylcholine_lvl_ * (1.0f - stress_level * 0.5f));
        }
    }

    [[nodiscard]] AutonomicMode get_mode() const { return current_mode_; }
    [[nodiscard]] float get_ach_level() const { return acetylcholine_lvl_; }
    [[nodiscard]] float get_dopamine_level() const { return dopamine_rpe_lvl_; }
    [[nodiscard]] bool is_parasympathetic() const { return current_mode_ == AutonomicMode::Parasympathetic; }
};
