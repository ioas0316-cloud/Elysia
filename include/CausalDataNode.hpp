#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

struct ObservationalFrame {
    float2 rotor_state;      // 클리포드/복소수 위상 로터 (cos θ, sin θ)
    float  curvature;        // 특이점 우회로 인한 공간 곡률 (κ)
    int    phase_space_type; // 0: ℝ (Real), 1: ℂ (Complex), 2: Cℓ₃,₀ (Clifford)
};

struct CausalMeaningContext {
    uint64_t    concept_id;          // 데이터의 본질적 개념 식별자
    float       semantic_energy;     // 지식 데이터가 지닌 실질적 정보량 (규격화 에너지)
    std::string causal_relation_tag; // 인과적 의미 태그
};

class CausalDataNode {
private:
    CausalMeaningContext meaning_; // 실체 (Data/Meaning)
    ObservationalFrame   frame_;   // 관측 도구 (Mathematical Frame)

public:
    CausalDataNode(uint64_t id, float energy, const std::string& tag) {
        meaning_ = {id, energy, tag};
        frame_   = {make_float2(1.0f, 0.0f), 0.0f, 0}; // 초기 1D 실수 관측 프레임
    }

    void shift_observational_frame(float2 new_rotor, float new_curvature, int new_space_type) {
        frame_.rotor_state = new_rotor;
        frame_.curvature = new_curvature;
        frame_.phase_space_type = new_space_type;

        std::cout << "[CausalDataNode] Meaning ID [" << meaning_.concept_id << "] (" << meaning_.causal_relation_tag << ")\n"
                  << " -> Reframed via Math Tool: Space Type [" << frame_.phase_space_type << "]"
                  << " | Curvature: " << frame_.curvature << "\n";
    }

    [[nodiscard]] const CausalMeaningContext& get_meaning() const { return meaning_; }
    [[nodiscard]] const ObservationalFrame& get_frame() const { return frame_; }
};
