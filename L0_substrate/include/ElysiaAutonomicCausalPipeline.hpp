#pragma once

#include "AutonomicStateController.hpp"
#include "CausalPhaseTensor.hpp"
#include <memory>

class ElysiaAutonomicCausalPipeline {
private:
    AutonomicStateController autonomic_ctrl_;
    std::unique_ptr<CausalPhaseTensor> phase_tensor_;

    cudaStream_t stream_realtime_{nullptr};
    cudaStream_t stream_background_{nullptr};

public:
    ElysiaAutonomicCausalPipeline(size_t num_elements) {
        phase_tensor_ = std::make_unique<CausalPhaseTensor>(num_elements);
        cudaStreamCreate(&stream_realtime_);
        cudaStreamCreate(&stream_background_);
    }

    ~ElysiaAutonomicCausalPipeline() {
        if (stream_realtime_) cudaStreamDestroy(stream_realtime_);
        if (stream_background_) cudaStreamDestroy(stream_background_);
    }

    void step(const EngineMetrics& metrics, float gradient_stagnation) {
        // 1. 자율 신경 계통(Sympathetic / Parasympathetic) 상태 업데이트
        autonomic_ctrl_.update(metrics);

        // 2. 신경 모드와 결합된 대수 공간(ℝ -> ℂ -> Cℓ₃,₀) 상전이 평가
        float effective_mismatch = metrics.mismatch_error * (1.0f + autonomic_ctrl_.get_ach_level() * 0.5f);
        phase_tensor_->evaluate_phase_transition(gradient_stagnation, effective_mismatch);

        // 3. 특이점(NaN/Inf) 발생 시 기하학적 회전 우회 커널 실행
        phase_tensor_->resolve_singularities(stream_realtime_);

        // 4. 신경 모드별 비동기 스트림 파이프라인 분기
        if (autonomic_ctrl_.is_parasympathetic()) {
            // [부교감 모드] 백그라운드 스트림에서 정화 및 위상 고착 진행
        } else {
            // [교감 모드] 실시간 스트림에서 노이즈 탐색 및 고차원 로터 스핀 연산 수행
        }
    }

    CausalPhaseTensor* get_phase_tensor() { return phase_tensor_.get(); }
    AutonomicStateController& get_autonomic_controller() { return autonomic_ctrl_; }
    cudaStream_t get_realtime_stream() const { return stream_realtime_; }
    cudaStream_t get_background_stream() const { return stream_background_; }
};
