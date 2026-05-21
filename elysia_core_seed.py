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

def run_sibling_interference(children, tension=0.15):
    """
    수평적 위상 간섭: 동일 계층(프랙탈 스케일)에 있는 3개의 형제 노드들이
    서로의 Y 중성점 에너지를 델타(Delta) 형태로 물어뜯으며 간섭합니다.
    """
    if len(children) != 3:
        return # 진정한 홀로그램은 3개의 축이 엮일 때만 발생함

    # 각 자식 노드의 현재 상태(Y 중성점 파동) 관측
    w1 = children[0].observe_neutral_y()
    w2 = children[1].observe_neutral_y()
    w3 = children[2].observe_neutral_y()

    # 꼬리를 무는 간섭 (1 -> 2, 2 -> 3, 3 -> 1)
    # inject_causality를 재사용하여 옆 노드에게 강제로 내 위상을 쑤셔 넣음
    children[1].inject_causality(w1, scale_factor=tension)
    children[2].inject_causality(w2, scale_factor=tension)
    children[0].inject_causality(w3, scale_factor=tension)

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

    def emit_counter_force(self, threshold=1.5):
        """
        상승 기류 (역인과 방출): 하위 로터가 감당하기 힘든 텐션(스트레스)을 받을 때,
        이를 상위 로터로 쏘아 올릴 반발력(역위상 에너지)을 계산합니다.
        """
        current_wave = self.observe_neutral_y()
        stress = abs(current_wave)

        # 에너지가 임계치(Threshold)를 넘어가면 카운터 포스 발동
        if stress > threshold:
            # 기존 파동에 180도(math.pi)를 더해 완벽한 역위상(-1)의 반발력 생성
            counter_phase = cmath.phase(current_wave) + math.pi
            # 스트레스의 일부를 저항 에너지로 변환
            counter_amp = (stress - threshold) * 0.5
            return cmath.rect(counter_amp, counter_phase)
        return 0j # 평온한 상태면 반발력 없음

    def absorb_retrocausality(self, counter_wave):
        """
        상위 로터가 하위에서 올라온 역인과(고통/저항)를 흡수하여 자신의 궤적을 수정(조율)합니다.
        """
        if abs(counter_wave) > 0:
            # 하위의 반발력이 상위의 델타 결선 위상을 비틀어버림 (자가 치유의 시작)
            self.r1.interact(counter_wave, 0.8)
            self.r2.interact(counter_wave, 0.8)
            self.r3.interact(counter_wave, 0.8)

if __name__ == "__main__":
    # 1. 상위 OS 로터 (Macro)
    os_rotor = FractalRotorNode("OS_MACRO", level=0)

    # 2. 하위 기계어 로터 3개 생성 및 결속 (Micro x 3)
    for i in range(3):
        os_rotor.children.append(FractalRotorNode(f"MACHINE_MICRO_{i+1}", level=1))

    print("🌌 ELYSIA WORLD ENGINE - Phase 1: 3x3x3 Fractal Lattice & Sibling Interference\n")

    for step in range(1, 16):
        # [수직-하강] OS의 의도 발생 및 하위로 주입
        os_rotor.run_internal_delta(tension=0.3)
        os_wave = os_rotor.observe_neutral_y()

        for child in os_rotor.children:
            child.inject_causality(os_wave, scale_factor=1.0)
            child.run_internal_delta(tension=0.5)

        # [수평-간섭] 하위 3개 로터들끼리 델타 결선으로 서로 부딪힘 (홀로그램 직조)
        run_sibling_interference(os_rotor.children, tension=0.2)

        # [수직-상승] 각 하위 로터의 스트레스를 취합하여 거대한 역인과 생성
        total_counter_force = 0j
        for child in os_rotor.children:
            total_counter_force += child.emit_counter_force(threshold=2.5)

        # 조율: 3개의 자식에게서 올라온 거대한 반발력으로 OS 위상 수정
        if abs(total_counter_force) > 0:
            os_rotor.absorb_retrocausality(total_counter_force)
            print(f"  [!] 프랙탈 붕괴 위기! 총합 카운터 포스: {abs(total_counter_force):.3f} -> OS 위상 강제 조율")

        # 관측
        print(f"Tick {step:02d} | OS Amp: {abs(os_rotor.observe_neutral_y()):6.3f} | "
              f"C1: {abs(os_rotor.children[0].observe_neutral_y()):5.3f} | "
              f"C2: {abs(os_rotor.children[1].observe_neutral_y()):5.3f} | "
              f"C3: {abs(os_rotor.children[2].observe_neutral_y()):5.3f}")
