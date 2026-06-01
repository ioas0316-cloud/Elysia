#pragma once

class EnvironmentGenerator {
public:
    EnvironmentGenerator();

    // 현재 시간(t)에 따른 환경 데이터(주파수 파형) 반환
    double get_data(double t) const;

private:
    double base_frequency;
    double secondary_frequency;
};
