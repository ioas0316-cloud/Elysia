import cmath
import math

class RotorVariable:
    def __init__(self, name, init_amp=1.0, init_phase=0.0):
        self.name = name
        self.amplitude = init_amp  # 데이터의 크기 (기존의 '값')
        self.phase = init_phase    # 위상 (0 ~ 2π)

    @property
    def wave_state(self):
        """현재 상태를 복소수(2D 파동 궤적)로 반환"""
        return cmath.rect(self.amplitude, self.phase)

    def interact(self, incoming_wave, tension):
        """외부 파동과 간섭하여 자신의 상태를 업데이트 (대입 연산자 = 의 대체)"""
        # 내 파동과 들어온 파동의 중첩
        new_state = self.wave_state + (incoming_wave * tension)

        # 간섭 결과로 새로운 진폭과 위상이 결정됨
        self.amplitude = abs(new_state)
        self.phase = cmath.phase(new_state)

def run_delta_cycle(r1, r2, r3, coupling_tension=0.1):
    """3개의 로터가 델타 결선으로 서로 간섭하며 에너지를 교환"""
    # 각 로터의 현재 파동 궤적 추출
    w1, w2, w3 = r1.wave_state, r2.wave_state, r3.wave_state

    # 서로 꼬리를 물고 간섭 (운동성이 운동성을 낳음)
    r2.interact(w1, coupling_tension)
    r3.interact(w2, coupling_tension)
    r1.interact(w3, coupling_tension)

def observe_y_neutral(r1, r2, r3):
    """와이 결선: 세 위상을 중성점(Zero)으로 모아 하나의 값으로 붕괴"""
    neutral_wave = (r1.wave_state + r2.wave_state + r3.wave_state) / 3.0

    # 파동이 스칼라(실수)로 확정되는 순간 (관측/해독)
    collapsed_value = abs(neutral_wave) * math.cos(cmath.phase(neutral_wave))
    return collapsed_value

if __name__ == "__main__":
    # 3개의 변수(로터) 생성
    node_A = RotorVariable("A", init_amp=10.0, init_phase=0.0)
    node_B = RotorVariable("B", init_amp=5.0,  init_phase=math.pi / 3) # 60도 위상차
    node_C = RotorVariable("C", init_amp=2.0,  init_phase=math.pi)     # 180도 위상차

    print("🌌 ELYSIA CORE SEED - Delta/Y Resonance Test\n")

    for step in range(1, 11):
        # 1. 연산: 델타 결선 안에서 서로 충돌하고 간섭
        run_delta_cycle(node_A, node_B, node_C, coupling_tension=0.2)

        # 2. 해독: 와이 결선을 통해 현재의 상태를 관측
        output = observe_y_neutral(node_A, node_B, node_C)

        print(f"Step {step:02d} | Neutral Output: {output:8.4f} | "
              f"A_Amp: {node_A.amplitude:5.2f}, B_Amp: {node_B.amplitude:5.2f}, C_Amp: {node_C.amplitude:5.2f}")
