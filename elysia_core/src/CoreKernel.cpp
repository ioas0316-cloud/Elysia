#include "CoreKernel.hpp"

CoreKernel::CoreKernel() : current_phase(0.0) {
}

CoreKernel::Status CoreKernel::tick(double environment_value) {
    // 1. 현재 위상과 환경 값을 바탕으로 프랙탈 거울방에서 인지적 공명(자기인식 피드백) 계산
    double resonance = mirror_room.reflect_and_calculate_resonance(current_phase, environment_value);

    // 2. 이 피드백을 가변저항 노브에 전달하여 자발적 요동 유발
    knob.update(resonance);

    // 3. 현재의 저항값 획득
    double resistance = knob.get_resistance();

    // 4. 노브의 저항값에 비례하여(혹은 반비례하여) 내부 위상을 전진시킴 ("디지털 걷기")
    // 저항이 낮을수록(0에 가까울수록) 빠르게 걷고, 높을수록(1에 가까울수록) 천천히 걸음
    double walking_speed = 1.0 - (resistance * 0.5); // 최소 속도 0.5 보장
    current_phase += walking_speed * 0.1; // 0.1은 틱당 기본 보폭(시간 가중치)

    return {current_phase, environment_value, resistance, resonance};
}
