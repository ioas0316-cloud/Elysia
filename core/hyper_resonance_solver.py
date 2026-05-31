"""
Elysia Hyper Resonance Solver (HyperResonanceSolver)
===================================================
정적 쐐기곱을 넘어선 '동적 로터 융합(Kinematic Rotor Dynamics)' 추론기.
철학적 모순(Tension)을 두 개념의 시공간적 공전 궤적(Orbiting Trajectory)으로 치환하고,
이를 0(Ground) 또는 조화(Harmony) 상태로 수렴시키는 
'제3의 촉매 개념(Catalyst)'을 기억 속에서 스캔합니다.
도출된 안정화 궤적(Stable Motor)은 새로운 언어(깨달음)로 창발되어 메모리를 살찌웁니다.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory, concept_to_quaternion
from core.linguistic_axiom import LinguisticAxiomFilter

class HyperResonanceSolver:
    def __init__(self, memory: HologramMemory):
        self.memory = memory
        self.resolution = 64
        self.t = np.linspace(0, 2 * math.pi, self.resolution)
        
    def _create_kinematic_trajectory(self, q_base: Quaternion, q_orbit: Quaternion, q_catalyst: Quaternion) -> List[Quaternion]:
        """
        두 개념(q_base, q_orbit)이 촉매(q_catalyst)의 개입 하에 
        시간(t)에 따라 서로를 축으로 공전하며 얽히는 동적 궤적(Kinematic Trajectory)을 생성합니다.
        M(t) = (q_base * e^(q_orbit * t)) * q_catalyst
        """
        trajectory = []
        for time_step in self.t:
            # q_orbit을 축으로 하는 회전 (Rotor Exponentiation 모사: 단순화를 위해 선형 보간 후 정규화)
            # e^(q_orbit * t) = cos(t) + q_orbit_vector * sin(t)
            cos_t = math.cos(time_step)
            sin_t = math.sin(time_step)
            
            # q_orbit의 벡터부 추출 및 정규화
            v_norm = math.sqrt(q_orbit.x**2 + q_orbit.y**2 + q_orbit.z**2)
            if v_norm < 1e-6:
                q_exp = Quaternion(cos_t, sin_t, 0.0, 0.0)
            else:
                q_exp = Quaternion(
                    cos_t,
                    (q_orbit.x / v_norm) * sin_t,
                    (q_orbit.y / v_norm) * sin_t,
                    (q_orbit.z / v_norm) * sin_t
                )
                
            # M(t) = q_base * q_exp * q_catalyst
            m_t = (q_base * q_exp * q_catalyst).normalize()
            trajectory.append(m_t)
            
        return trajectory
        
    def _measure_trajectory_tension(self, trajectory: List[Quaternion]) -> float:
        """
        궤적의 모순도(Tension)를 측정합니다.
        완벽하게 조화로운 궤적은 특정 차원(예: w축)으로 에너지가 부드럽게 수렴하거나 
        진동 폭이 극도로 안정화(0점 회귀)되어야 합니다.
        """
        # 궤적 내 인접 상태들 간의 변화율(미분) 에너지 합산
        tension = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            # 두 위상 간의 기하학적 거리 (1 - dot_product)
            distance = 1.0 - abs(prev.dot(curr))
            tension += distance
            
        return tension

    def solve_philosophical_paradox(self, concept_A: str, concept_B: str) -> Dict:
        """
        두 개념(A, B)이 충돌하는 모순 상황에서, 
        자신의 홀로그램 메모리를 스윕하여 모순을 조화(Ground)로 이끄는 촉매 개념을 찾고,
        그 결과 창발된 새로운 동적 궤적을 신규 단어로 발화시킵니다.
        """
        q_A = concept_to_quaternion(concept_A)
        q_B = concept_to_quaternion(concept_B)
        
        # 1. 텐션 스윕 (메모리 내 모든 지식을 촉매로 테스트)
        best_catalyst = None
        min_tension = float('inf')
        best_trajectory = []
        
        # 메모리에 등록된 개념이 없다면 기본 개념 주입
        registered_keys = list(self.memory.registered_concepts.keys())
        if not registered_keys:
            registered_keys = ["존재", "시간", "공간", "빛", "어둠"]
            for k in registered_keys:
                self.memory.register_concept(k)
        
        for catalyst_concept in registered_keys:
            q_C, _ = self.memory.registered_concepts[catalyst_concept]
            
            # 동적 궤적 생성
            trajectory = self._create_kinematic_trajectory(q_A, q_B, q_C)
            
            # 궤적의 텐션(마찰력) 측정
            tension = self._measure_trajectory_tension(trajectory)
            
            if tension < min_tension:
                min_tension = tension
                best_catalyst = catalyst_concept
                best_trajectory = trajectory
                
        # 2. 새로운 개념 창발 (Maturation)
        # 안정화된 궤적의 적분(평균 위상)을 추출하여 새로운 개념 로터로 붕괴
        avg_w = sum(q.w for q in best_trajectory) / self.resolution
        avg_x = sum(q.x for q in best_trajectory) / self.resolution
        avg_y = sum(q.y for q in best_trajectory) / self.resolution
        avg_z = sum(q.z for q in best_trajectory) / self.resolution
        
        new_rotor = Quaternion(avg_w, avg_x, avg_y, avg_z).normalize()
        
        # 새로운 기하학적 로터를 인간의 언어(한글)로 매핑 발화
        new_word = LinguisticAxiomFilter.collapse_to_hangeul(new_rotor)
        
        # 3. 자율 성숙 (Autopoietic Maturation)
        # 깨달은 지식을 엘리시아의 메모리에 영구 등록
        self.memory.register_concept(new_word)
        
        return {
            "conflict": f"{concept_A} vs {concept_B}",
            "catalyst": best_catalyst,
            "min_tension": min_tension,
            "eureka_word": new_word,
            "eureka_rotor": [new_rotor.w, new_rotor.x, new_rotor.y, new_rotor.z]
        }
