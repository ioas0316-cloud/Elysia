#pragma once

class FractalMirrorRoom {
public:
    FractalMirrorRoom();

    // 과거 위상과 현재 위상을 대조하여 시차(Delta)와 인지적 공명(Cognitive Resonance) 반환
    // current_phase: 현재 나의 위상
    // environment_value: 현재 관측된 외부 환경 값
    double reflect_and_calculate_resonance(double current_phase, double environment_value);

private:
    double previous_phase;
    double previous_environment_value;
    bool is_first_tick;
};
