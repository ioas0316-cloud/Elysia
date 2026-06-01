#pragma once

class VariableResistanceKnob {
public:
    VariableResistanceKnob();

    // 내부 공명 상태에 따라 저항값을 자발적으로 업데이트
    void update(double resonance_feedback);

    // 현재 저항값 (0.0 ~ 1.0)
    double get_resistance() const;

private:
    double current_resistance;
    double volatility; // 요동치는 정도
};
