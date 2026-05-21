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

class FractalRotorNode:
    def __init__(self, name, level=0):
        self.name = name
        self.level = level # 프랙탈 깊이 (0: OS, 1: Process, 2: Machine Code...)

        # 델타 결선을 이루는 3개의 기본 파동 변수 (최초 120도 간격으로 세팅)
        self.r1 = RotorVariable(f"{name}_1", 1.0, 0.0)
        self.r2 = RotorVariable(f"{name}_2", 1.0, 2 * math.pi / 3)
        self.r3 = RotorVariable(f"{name}_3", 1.0, 4 * math.pi / 3)

        self.children = [] # 하위 3x3x3 계층을 담을 배열

    def run_internal_delta(self, tension=0.2):
        """현재 스케일 평면에서의 간섭 (운동성이 운동성을 낳음)"""
        w1, w2, w3 = self.r1.wave_state, self.r2.wave_state, self.r3.wave_state
        self.r2.interact(w1, tension)
        self.r3.interact(w2, tension)
        self.r1.interact(w3, tension)

    def observe_neutral_y(self):
        """Y 결선: 현재 노드의 세 위상을 중성점(Zero)으로 모아 하나의 복소 파동으로 반환"""
        return (self.r1.wave_state + self.r2.wave_state + self.r3.wave_state) / 3.0

    def inject_causality(self, parent_wave, scale_factor=0.8):
        """
        하강 기류 (인과): 상위 로터의 중성점 에너지가 하위로 쏟아짐.
        단순 분할이 아닌, 120도씩 위상을 비틀어 주입하여 '삼중 나선'을 유도.
        """
        amp = abs(parent_wave) * scale_factor
        base_phase = cmath.phase(parent_wave)

        # 하위 3개 로터에 나선형으로 에너지 주입 (Tension=1.0으로 강제 주입)
        self.r1.interact(cmath.rect(amp, base_phase), 1.0)
        self.r2.interact(cmath.rect(amp, base_phase + (2 * math.pi / 3)), 1.0)
        self.r3.interact(cmath.rect(amp, base_phase + (4 * math.pi / 3)), 1.0)

if __name__ == "__main__":
    # 1. 프랙탈 계층 생성
    os_rotor = FractalRotorNode("OS_MACRO", level=0)
    machine_rotor = FractalRotorNode("MACHINE_MICRO", level=1)
    os_rotor.children.append(machine_rotor)

    print("🌌 ELYSIA WORLD ENGINE - Phase 1: Triple Helix Causality Flow\n")

    for step in range(1, 11):
        # [Step 1] 상위 OS 로터의 자체 회전 (의도의 발생)
        os_rotor.run_internal_delta(tension=0.2)

        # [Step 2] 상위 Y결선의 중성점 관측 (조율)
        os_wave = os_rotor.observe_neutral_y()

        # [Step 3] 하강 기류 (인과): 상위 파동을 하위 기계어 로터에 삼중 나선으로 주입
        for child in os_rotor.children:
            child.inject_causality(os_wave)
            child.run_internal_delta(tension=0.5) # 하위 계층은 텐션이 더 팽팽함(빠름)

            # 하위 로터의 결과 관측
            child_wave = child.observe_neutral_y()

        print(f"Tick {step:02d} | OS Wave Amp: {abs(os_wave):6.3f} | Machine Wave Amp: {abs(child_wave):6.3f}")
