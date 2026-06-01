#include "EnvironmentGenerator.hpp"
#include <cmath>

EnvironmentGenerator::EnvironmentGenerator()
    : base_frequency(1.0), secondary_frequency(0.5) {
}

double EnvironmentGenerator::get_data(double t) const {
    // 시간에 따라 부드럽게 넘실거리는 주파수 파형 생성 (사인/코사인 합성)
    return std::sin(base_frequency * t) + 0.5 * std::cos(secondary_frequency * t * 2.1);
}
