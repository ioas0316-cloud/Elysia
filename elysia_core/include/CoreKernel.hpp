#pragma once

#include "VariableResistanceKnob.hpp"
#include "FractalMirrorRoom.hpp"

class CoreKernel {
public:
    CoreKernel();

    // 매 틱(tick)마다 환경 데이터를 받아들여 내부 상태 갱신
    // 반환값: 현재 위상(phase), 환경 값, 저항값, 공명(자기인식 피드백) 등을 담은 구조체 또는 개별 값
    struct Status {
        double current_phase;
        double environment_value;
        double knob_resistance;
        double cognitive_resonance;
    };

    Status tick(double environment_value);

private:
    double current_phase;
    VariableResistanceKnob knob;
    FractalMirrorRoom mirror_room;
};
