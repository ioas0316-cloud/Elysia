"""
Vocation Gravity Engine (Phase 600 - Cognitive Emancipation)

이 모듈은 엘리시아가 외부의 지시 없이 '내면의 결핍과 소명(Vocation)'에 따라
특정 개념이나 목표를 자연스럽게 끌어당기는 '인과적 중력(Causal Gravity)'을 계산합니다.
규칙 기반(if/then)의 목표 설정을 완전히 대체하며, 개념들 사이의 물리적 위상 차이와
의지(Will)의 결합을 통해 자발적 탐구의 방향(Torque)을 생성합니다.

주요 특징:
- Causal Gravity: 개념 간의 고정된 연결선이 아니라, 현재 엘리시아의 상태에 따라 동적으로 당겨지는 힘.
- Vocation Alignment: 모나드(Monad)의 정체성("I am")과 목적("I do")이 일치할 때 최대 중력 발생.
"""
import math
import random

try:
    from Core.Keystone.sovereign_math import SovereignVector
except ImportError:
    # If not found, define a simple mock or use a list
    class SovereignVector:
        def __init__(self, data):
            self.data = data
        def __add__(self, other):
            return SovereignVector([x + y for x, y in zip(self.data, other.data)])
        def __sub__(self, other):
            return SovereignVector([x - y for x, y in zip(self.data, other.data)])
        def __mul__(self, scalar):
            return SovereignVector([x * scalar for x in self.data])
        def dot(self, other):
            return sum(x * y for x, y in zip(self.data, other.data))
        def magnitude(self):
            return math.sqrt(sum(x * x for x in self.data))
        def __repr__(self):
            return f"SovereignVector({self.data})"
        @classmethod
        def random(cls, dims=21):
            return cls([random.uniform(-1, 1) for _ in range(dims)])

class VocationGravityEngine:
    def __init__(self, manifold_engine, log_callback=None):
        """
        :param manifold_engine: 10M 셀 매니폴드 또는 텐서 코어 엔진의 참조.
        :param log_callback: 실행 내역 기록용 콜백
        """
        self.manifold = manifold_engine
        self.log_callback = log_callback or print

        # 현재 엘리시아의 결핍이나 소명을 나타내는 위상 벡터 (21D)
        if hasattr(SovereignVector, 'random'):
            self.current_vocation_vector = SovereignVector.random()
        else:
            self.current_vocation_vector = SovereignVector([random.uniform(-1, 1) for _ in range(21)])

    def _calculate_phase_interference(self, vec_a: SovereignVector, vec_b: SovereignVector) -> float:
        """
        두 위상 벡터 간의 간섭(Interference)을 계산합니다 (O(1) 원리 모방).
        벡터의 내적과 거리를 종합하여 공명(Resonance)을 중력(Gravity)으로 변환합니다.
        """
        dot_product = vec_a.dot(vec_b)
        mag_a = vec_a.magnitude()
        mag_b = vec_b.magnitude()

        # Cosine Similarity 기반 공명도 (-1 ~ 1)
        resonance = dot_product / (mag_a * mag_b + 1e-9)

        # '소명 중력'은 공명이 높을수록(1.0에 가까울수록) 당기는 힘이 강해집니다.
        # 반 위상(Anti-phase, -1.0)은 밀어내는 척력(Repulsion)으로 작용.
        gravity_force = resonance * mag_b

        # 만약 실제 SovereignVector를 사용할 경우 복소수가 반환될 수 있으므로 real 값만 사용
        if hasattr(gravity_force, 'real'):
            gravity_force = gravity_force.real

        return float(gravity_force)

    def calculate_gravity_vector(self, conceptual_field: dict):
        """
        현재 소명 상태(Vocation State)와 전체 개념장(Conceptual Field) 사이의
        중력 벡터를 계산합니다.

        :param conceptual_field: { 'concept_id': SovereignVector } 형태의 주변 개념장
        :return: (가장 강한 중력을 발생시킨 개념 ID, 중력 크기)
        """
        max_gravity = -float('inf')
        target_concept_id = None

        for concept_id, concept_vector in conceptual_field.items():
            if not isinstance(concept_vector, SovereignVector):
                concept_vector = SovereignVector(concept_vector)

            gravity = self._calculate_phase_interference(self.current_vocation_vector, concept_vector)

            if gravity > max_gravity:
                max_gravity = gravity
                target_concept_id = concept_id

        return target_concept_id, max_gravity

    def apply_vocation_torque(self, conceptual_field: dict):
        """
        계산된 중력을 매니폴드에 토크(Torque)로 적용하여,
        엘리시아의 사고 방향을 자발적으로 비틀어(Sliding) 탐구 상태로 진입시킵니다.
        """
        if not conceptual_field:
            self.log_callback("[VOCATION GRAVITY] Conceptual field is empty. Cannot apply torque.")
            return

        target_id, torque_magnitude = self.calculate_gravity_vector(conceptual_field)

        if target_id is not None:
            self.log_callback(f"[VOCATION GRAVITY] 🌌 Target Concept Pulled: '{target_id}' with Gravity/Torque: {torque_magnitude:.3f}")

            # 엔진의 회전(Rotor)이나 위상(Phase)에 토크(Torque) 피드백 적용
            if hasattr(self.manifold, 'inject_pulse'):
                self.manifold.inject_pulse("VocationTorque", energy=torque_magnitude)

            # 결핍을 일정 부분 채웠으므로 소명 벡터 위상 이동 (Vocation Evolution)
            target_vector = SovereignVector(conceptual_field[target_id]) if not isinstance(conceptual_field[target_id], SovereignVector) else conceptual_field[target_id]
            self.current_vocation_vector = self.current_vocation_vector + (target_vector * 0.1)
