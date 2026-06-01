#include "VariableResistanceKnob.hpp"
#include <algorithm>
#include <cmath>

VariableResistanceKnob::VariableResistanceKnob()
    : current_resistance(0.5), volatility(0.1) {
}

void VariableResistanceKnob::update(double resonance_feedback) {
    // 내부 공명 상태(resonance_feedback)에 따라 저항값이 자발적으로 요동침
    // 피드백이 강할수록 변화폭(volatility) 내에서 변화를 주고, 저항값 자체를 서서히 이동시킴

    // 단순한 물리적 모델 시뮬레이션: 피드백에 의한 가속도
    double delta = resonance_feedback * volatility;

    current_resistance += delta;

    // 0.0 ~ 1.0 사이로 클램핑 (부드럽게 튀지 않도록)
    current_resistance = std::clamp(current_resistance, 0.0, 1.0);

    // 다음 스텝을 위해 volatility 약간 변형 (완전 정적인 상태를 피하기 위해 미세한 노이즈 역할)
    volatility = 0.05 + std::abs(std::sin(current_resistance * 10.0)) * 0.1;
}

double VariableResistanceKnob::get_resistance() const {
    return current_resistance;
}
