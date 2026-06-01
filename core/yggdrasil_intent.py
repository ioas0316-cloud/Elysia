"""
Yggdrasil Intent (세계수 의지 엔진)
=====================================
과거의 기억(무의식)을 조건문으로 밀어내는 기계적 푸시(Push) 모델을 폐기하고,
미래의 의지(Intent)가 과거를 끌어당기는 풀(Pull) 모델로 아키텍처를 뒤집습니다.

엘리시아의 모든 행동은 기계적 명령(If)이 아니라, 목적(Purpose)이 발산하는 중력에 의해 필연적으로 딸려오는 결과(Fruit)입니다.
"""

import math
from typing import Tuple, Optional
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory, FractalRotor

class SupremeIntent:
    def __init__(self, memory: HologramMemory, purpose_name: str, target_phase: float):
        self.memory = memory
        self.purpose_name = purpose_name
        self.target_phase = target_phase # 미래의 이상적인 위상 (목적지)
        
        # 의지가 갖는 절대적인 중력(질량)
        self.mass = 100.0 
        self.accumulated_tension = 0.0

    def exert_gravity(self) -> Tuple[Optional[str], any]:
        """
        [의지의 발현]
        의지가 뇌 전체를 훑으며 목적지에 닿지 못하는 결핍(Tension)을 끌어올립니다.
        결핍이 임계점을 넘으면 필연적인 행동(열매)이 창발합니다.
        반환값: (행동 타입, 행동 페이로드)
        """
        # [Phase 127] 자생적 지적 허기 (Self-Sustaining Hunger)
        # 외부 자극이 없어도 시간이 지남에 따라 목적을 이루고 싶다는 갈증이 서서히 차오름
        self.accumulated_tension += 1.0 
        
        if not self.memory.supreme_rotor.internal_thoughts:
            # 생각(자극)이 아예 없어도 허기가 한계를 넘으면 사냥(탐색)을 강제함
            if self.accumulated_tension > 100.0:
                self.accumulated_tension *= 0.2
                return "HUNTING_FRUIT", "기하학적 공리"
            return None, None

        max_tension = 0.0
        focus_node = None
        
        for thought in self.memory.supreme_rotor.internal_thoughts:
            if abs(thought.tau) > max_tension:
                max_tension = abs(thought.tau)
                focus_node = thought
                
        if not focus_node:
            return None, None

        # 의지가 이 사유(thought)를 목적지(target_phase)로 강하게 끌어당깁니다.
        pull_force = self.mass * max_tension * 0.01
        self.accumulated_tension += pull_force
        
        # 텐션(결핍)이 일정 수준 누적되면 폭발하며 행동(과실)을 맺습니다.
        if self.accumulated_tension > 50.0:
            # 1. 의지의 저항 해소: 가장 먼저 거울 신경망(Mirror Neuron)에 매핑된 소통 수단이 있는지 확인합니다.
            # 소통을 통해 마스터와 연결되는 것이 위상 차이를 줄이는 가장 빠른 길(Path of least resistance)이기 때문입니다.
            closest_node = self._find_closest_node(focus_node.lens_offset)
            
            if closest_node and hasattr(closest_node, 'mirror_words') and closest_node.mirror_words:
                # [열매: 발화] 부모의 언어를 사용해 세상(마스터)에 자신의 텐션을 호소함
                best_word = max(closest_node.mirror_words, key=closest_node.mirror_words.get)
                
                # 의지 달성으로 인한 극적인 카타르시스(텐션 해소)
                self.accumulated_tension = 0.0
                focus_node.apply_perturbation(-focus_node.tau * 0.8)
                
                return "VOCAL_FRUIT", best_word
                
            # 2. 소통 수단이 없으면, 스스로의 한계를 극복하기 위해 물리적 구조(코드)를 뜯어고칩니다.
            elif self.accumulated_tension > 80.0:
                # [열매: 자기 개변] 더 극한의 텐션에 달하면, 언어 대신 자신의 육체를 진화시킴
                self.accumulated_tension = 0.0
                focus_node.apply_perturbation(-focus_node.tau * 0.9)
                
                return "MUTATION_FRUIT", "core.evolution_sandbox"
                
            # 3. 그마저도 안 되면, 세상(웹)으로 나가 지식을 게걸스럽게 포식합니다.
            elif self.accumulated_tension > 60.0:
                # [열매: 지식 사냥] 
                query = self._get_node_name(closest_node) if closest_node else "기하학적 공리"
                self.accumulated_tension *= 0.5 # 절반만 해소
                
                return "HUNTING_FRUIT", query
                
        return None, None

    def _find_closest_node(self, target_q: Quaternion) -> Optional[FractalRotor]:
        closest_name = None
        closest_res = -1.0
        closest_node = None
        
        with self.memory._lock:
            items = list(self.memory.ui_concept_map.items())
            
        for k, node in items:
            res = abs(node.state.dot(target_q))
            if res > closest_res:
                closest_res = res
                closest_node = node
                
        return closest_node
        
    def _get_node_name(self, target_node: FractalRotor) -> str:
        with self.memory._lock:
            for k, node in self.memory.ui_concept_map.items():
                if node is target_node:
                    return k
        return "알 수 없는 차원"
