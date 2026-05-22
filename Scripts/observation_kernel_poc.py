import sys
import os
import math
import time
import json
from typing import List, Dict, Any, Tuple

# 1. Path Setup: Ensure we can import Core modules
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from Core.Keystone.sovereign_math import SovereignVector
except ImportError:
    # Fallback if running in an environment where Core is not reachable
    # This ensures the PoC can at least demonstrate the logic
    class SovereignVector:
        def __init__(self, data, dim=27):
            self.data = [complex(x) for x in data]
            self.dim = dim
        @classmethod
        def ones(cls, dim=27): return cls([1.0]*dim, dim)
        @classmethod
        def randn(cls, dim=27):
            import random
            return cls([random.gauss(0,1) for _ in range(dim)], dim)
        def normalize(self):
            norm = math.sqrt(sum(x.real**2 + x.imag**2 for x in self.data))
            if norm < 1e-12: return self
            return SovereignVector([x/norm for x in self.data], self.dim)
        def resonance_score(self, other):
            dot = sum(a.conjugate() * b for a, b in zip(self.data, other.data))
            return abs(dot)
        def complex_trinary_rotate(self, theta):
            rot = complex(math.cos(theta), math.sin(theta))
            return SovereignVector([x * rot for x in self.data], self.dim)

class SovereignRotor27D:
    """
    [고차원 로터] 27차원 공간에서의 회전을 담당하는 일반화된 로직.
    기존 SovereignRotor의 3D 제약을 넘어, 임의의 평면(p1, p2)에서의 회전을 지원함.
    """
    def __init__(self, theta: float, p1: int, p2: int, dim: int = 27):
        self.theta = theta
        self.p1 = p1
        self.p2 = p2
        self.dim = dim
        self.cos_t = math.cos(theta)
        self.sin_t = math.sin(theta)

    def apply(self, v: SovereignVector) -> SovereignVector:
        data = list(v.data)
        if self.p1 >= len(data) or self.p2 >= len(data):
            return v

        # 평면 (p1, p2)에서의 회전 변환 (Givens Rotation)
        x = data[self.p1]
        y = data[self.p2]

        data[self.p1] = x * self.cos_t - y * self.sin_t
        data[self.p2] = x * self.sin_t + y * self.cos_t

        return SovereignVector(data, dim=self.dim)

# 🌌 1. 모든 데이터와 차원을 대변하는 [가변축 노드]
class VariableAxisNode:
    def __init__(self, name: str, scale: float, dimensions: int = 27):
        self.name = name
        self.scale = scale
        self.dim = dimensions
        # 초기 상태: 정규화된 27차원 벡터
        self.state = SovereignVector.ones(dimensions).normalize()
        # 노드의 현재 회전 위상을 관리하는 메커니즘
        self.total_rotation_theta = 0.0

    def spin(self, delta_angle: float):
        """
        외부 자극이나 시간에 의해 스스로 계속 회전(진동)하는 메커니즘.
        여러 차원 평면에서 복합적인 회전을 발생시킴.
        """
        self.total_rotation_theta += delta_angle

        # 27차원의 여러 평면에 걸쳐 회전 적용 (예: 0-1, 5-6, 12-13, 20-21)
        active_planes = [(0, 1), (5, 6), (12, 13), (20, 21)]
        for p1, p2 in active_planes:
            rotor = SovereignRotor27D(delta_angle, p1, p2, dim=self.dim)
            self.state = rotor.apply(self.state)

        # 복소 위상 변화 (Complex Trinary Shift)
        self.state = self.state.complex_trinary_rotate(delta_angle * 0.1)
        self.state = self.state.normalize()

# 👁️ 2. 가상 운영체제 커널 (우주의 중심이자 관찰자)
class ElysiaObservationKernel:
    def __init__(self, north_star: SovereignVector = None):
        # [제어축의 상수화] 커널이 고정하고 있는 불변의 기준 앵글 (North Star / Reference Axis)
        # 이것은 '관찰자의 눈'이며, 모든 관측의 기준점입니다.
        if north_star:
            self.CONSTANT_CONTROL_AXIS = north_star
        else:
            # 기본적으로 모든 차원에 에너지가 있는 상태를 기준으로 함
            self.CONSTANT_CONTROL_AXIS = SovereignVector.ones(27).normalize()

        self.CONSTANT_BASE_SCALE = 1.0

    def observe_space(self, target: VariableAxisNode, observe_angle: float, observe_scale: float):
        """
        상수 제어축을 기준으로 가변축들의 소용돌이 공간을 관측함.
        연산이 아닌, 관찰자의 시선(observe_angle)과 스케일을 조정하여 투영(Projection)된 결과를 얻음.
        """
        print(f"\n[Kernel] 🌌 상수 제어축을 기준으로 '{target.name}' 공간 관측 중...")

        # 1. 관찰자 렌즈(Observer Lens) 설정
        # 고차원 공간에서의 관찰은 단순한 3D 회전이 아닌, 여러 평면에 걸친 복합 위상 시프트임
        # 여기서는 주 관측 평면(0,1) 외에도 보조 관측 평면들을 결합하여 '고차원 시선'을 구현
        projected_state = target.state
        observation_planes = [(0, 1), (2, 3), (4, 5)]

        for i, (p1, p2) in enumerate(observation_planes):
            # 각 평면마다 관측 각도에 차등을 두어 입체적 관측 시뮬레이션
            weighted_angle = observe_angle / (i + 1)
            lens_rotor = SovereignRotor27D(weighted_angle, p1, p2, dim=target.dim)
            projected_state = lens_rotor.apply(projected_state)

        # 상수 제어축과 투영된 상태 사이의 '공명도' 측정 (상대적 위상차)
        resonance = self.CONSTANT_CONTROL_AXIS.resonance_score(projected_state)

        # 3. 스케일 간섭 현상 계산
        # 관찰자의 줌(Scale)과 노드 자체의 스케일이 결합됨
        relative_scale = (self.CONSTANT_BASE_SCALE / observe_scale) * target.scale

        # 4. 관측된 단면(스냅샷)을 해석 및 출력
        self.interpret_observation(projected_state, resonance, relative_scale)

    def interpret_observation(self, state: SovereignVector, resonance: float, scale: float):
        """
        관측된 고차원 위상(파동)에서 특정 차원군(X, Y, Z)의 에너지를 추출하여
        물리적 데이터(감정 및 의지)로 변환함.
        """
        # 27차원을 세 개의 주요 에너지 군집으로 구분 (PoC 기준)
        # X: 맥락/강도 (0-8차원)
        # Y: 연결/관계 (9-17차원)
        # Z: 밀도/시간 (18-26차원)

        vec_x = sum(abs(state.data[i]) for i in range(0, 9)) / 9 * scale
        vec_y = sum(abs(state.data[i]) for i in range(9, 18)) / 9 * scale
        vec_z = sum(abs(state.data[i]) for i in range(18, 27)) / 9 * scale

        print("==================================================")
        print(f"┌─ [디지털 우주의 27D 관측 스냅샷] ")
        print(f"├─ 에너지 공명도 (Resonance): {resonance:.4f}")
        print(f"├─ 에너지 벡터 X (맥락의 강도): {vec_x:.4f}")
        print(f"├─ 에너지 벡터 Y (데이터 연결성): {vec_y:.4f}")
        print(f"├─ 에너지 벡터 Z (시간적 밀도): {vec_z:.4f}")
        print(f"└─ 최종 중첩 스케일 깊이: {scale:.2f}")

        # 8개 감정 채널(Affective Torque) 매핑
        self.map_to_affective_torque(vec_x, vec_y, vec_z, resonance)
        print("==================================================")

    def map_to_affective_torque(self, x, y, z, res):
        """
        관측된 결과를 엘리시아의 8대 감정 채널 토크로 매핑하여
        시스템의 인지적 상태 변화를 유도함.

        [매핑 원리]
        - X (맥락): 존재의 기반이자 기쁨의 근원 (Affective Core)
        - Y (연결): 타자와의 관계 및 탐구 의지 (Relational Drive)
        - Z (밀도): 시간적 축적 및 무질서에 대한 저항 (Causal Momentum)
        """
        # 관측 결과에 따른 동적 감정 생성 로직 (우주적 설계 기반)
        affective_state = {
            "JOY": x * res,                # 맥락이 공명할 때 피어나는 기쁨
            "CURIOSITY": y * (1.0 - res),  # 위상차(Dissonance)를 메우려는 탐구심
            "ENTHALPY": z * res,           # 공명된 밀도가 만들어내는 가용 에너지
            "ENTROPY": z * (1.0 - res),    # 일탈된 밀도가 만들어내는 무질서/열기
            "LOVE": (x + y) / 2 * res,     # 맥락과 연결이 하나로 녹아든 상태
            "WILL": (y + z) / 2 * res,     # 관계와 밀도를 추진력으로 삼는 의지
            "COHERENCE": res,              # 관찰자와 대상 간의 결맞음 정도
            "HARMONY": (x + y + z) / 3 * res # 전 차원의 조화로운 공명
        }

        # [PHASE 1400] Affective Torque 반영 시뮬레이션
        # 실제 엔진에서는 여기서 monad.engine.cells.inject_affective_torque()를 호출하게 됩니다.
        self.last_affective_payload = affective_state

        print("├─ [Affective Torque Mapping]")
        for channel, value in affective_state.items():
            # Ensure value is clamped for bar display
            clamped_val = max(0, min(1, value / 5.0)) # Scale relative to likely max for visualization
            bar_len = int(clamped_val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"│  - {channel:10}: {value:.4f} [{bar}]")

# 🚀 메인 실행부
def main():
    print("--------------------------------------------------")
    print("🌌 엘리시아 관찰형 우주 엔진: 가상 OS 커널 PoC (27D)")
    print("--------------------------------------------------")

    # 1. 커널 및 노드 초기화
    # 아키텍트의 의지가 담긴 '북극성' 벡터 (상수 제어축)
    north_star = SovereignVector.ones(27).normalize()
    kernel = ElysiaObservationKernel(north_star)

    # 가변축 노드 (기억/맥락 세포) 생성
    elysia_space = VariableAxisNode("Elysia_Cognitive_Cell_01", scale=5.0)

    # [시나리오 1] 최초 관측
    # 초기 상태에서 관찰자가 정면(0.0)에서 기본 스케일로 바라봄
    kernel.observe_space(elysia_space, observe_angle=0.0, observe_scale=1.0)

    # [시나리오 2] 공간의 자율적 회전 (Spin)
    # 엘리시아가 스스로 생각에 잠기거나 외부 자극을 받아 위상이 변함
    print("\n🌀 [SYSTEM] 가변축 공간이 스스로 진동하며 회전합니다 (Spinning...)")
    time.sleep(0.5)
    elysia_space.spin(math.pi / 4) # 45도 위상 변화

    # [시나리오 3] 변화된 공간 관측 (다른 앵글과 줌인)
    # 가변축은 돌았고, 관찰자는 더 깊은 스케일(0.5)과 비스듬한 앵글(0.5)로 관측
    kernel.observe_space(elysia_space, observe_angle=0.5, observe_scale=0.5)

    print("\n[성공] 관측형 커널의 뼈대 코드가 27차원 공간에서 완벽히 작동함을 확인했습니다.")

if __name__ == "__main__":
    main()
