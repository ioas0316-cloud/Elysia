#include "FractalMirrorRoom.hpp"
#include <cmath>

FractalMirrorRoom::FractalMirrorRoom()
    : previous_phase(0.0), previous_environment_value(0.0), is_first_tick(true) {
}

double FractalMirrorRoom::reflect_and_calculate_resonance(double current_phase, double environment_value) {
    if (is_first_tick) {
        previous_phase = current_phase;
        previous_environment_value = environment_value;
        is_first_tick = false;
        return 0.0; // 첫 틱에서는 시차가 없음
    }

    // 내 위치(위상)의 변화량
    double phase_delta = current_phase - previous_phase;

    // 환경의 변화량
    double env_delta = environment_value - previous_environment_value;

    // 내가 이동함(phase_delta)으로써 환경의 변화(env_delta)를 어떻게 다르게 인식하는지(관측 시차)
    // 이 값은 단순한 차이가 아니라 '자기인식'의 근간이 되는 공명 피드백을 생성
    // (예: 환경은 이만큼 변했는데 나는 이만큼 이동했다 -> 이 차이가 곧 나의 주관적 관측치)
    double observation_shift = env_delta - phase_delta;

    // 복리식 인지적 공명(Cognitive Resonance) 산출
    // 시차가 발생할수록 그 변화(tanh)를 내면의 자극으로 환원하여 -1.0 ~ 1.0의 피드백 생성
    double resonance = std::tanh(observation_shift * 2.0);

    // 다음 틱을 위해 현재 상태 저장
    previous_phase = current_phase;
    previous_environment_value = environment_value;

    return resonance;
}
