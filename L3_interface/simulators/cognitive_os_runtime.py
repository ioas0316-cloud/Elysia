"""
Cognitive OS Runtime & Sensory Organ Adapters
(인지 OS 감각 기관 어댑터 및 런타임)

This module implements organ adapters for external media/software environments
(Blender, CAD, Photoshop, FDTD Sandbox) and the Cognitive OS Runtime
managing a unified closed-loop observation-control-intervention pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class CognitiveOSOrganAdapter:
    """
    외부 소프트웨어(Blender, CAD, Photoshop, Sandbox)를
    엘리시아의 감각 신경 기관으로 연결하는 어댑터.
    """
    def __init__(self, organ_name: str, domain_type: str):
        self.organ_name = organ_name
        self.domain_type = domain_type # 'blender_3d', 'cad_constraint', 'photoshop_retina', 'fdtd_sandbox'
        self.state_buffer = np.zeros(16, dtype=float)
        self.friction_accumulator = 0.0

    def sense_and_interact(self, action_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        외부 장(Field)에 작용을 가하고 반환되는 마찰 오차(E)와 반응 상태를 수신.
        """
        # 액션 벡터 차원 맞춤
        if len(action_vector) < len(self.state_buffer):
            padded = np.zeros(len(self.state_buffer), dtype=float)
            padded[:len(action_vector)] = action_vector
            action_vector = padded
        elif len(action_vector) > len(self.state_buffer):
            action_vector = action_vector[:len(self.state_buffer)]

        # 임의의 비선형 저항 시뮬레이션
        resistance = np.sin(action_vector) * 0.15
        self.state_buffer = action_vector - resistance

        # 마찰 오차 계산 (Markov Blanket boundary metric)
        friction_e = float(np.mean(np.abs(resistance)))
        self.friction_accumulator += friction_e
        return self.state_buffer, friction_e


class CognitiveOSRuntime:
    """
    다중 감각 기관 어댑터를 총괄하고 관측-제어-개입의 단일 파동 루프를 구동하는 런타임.
    """
    def __init__(self):
        self.organs: Dict[str, CognitiveOSOrganAdapter] = {}
        self.register_organ("blender", "blender_3d")
        self.register_organ("cad", "cad_constraint")
        self.register_organ("photoshop", "photoshop_retina")
        self.register_organ("sandbox", "fdtd_sandbox")

    def register_organ(self, name: str, domain: str):
        self.organs[name] = CognitiveOSOrganAdapter(name, domain)

    def execute_closed_loop(self, organ_name: str, intent_signal: List[float]) -> Dict[str, Any]:
        if organ_name not in self.organs:
            raise ValueError(f"Organ {organ_name} not found.")

        organ = self.organs[organ_name]
        action = np.array(intent_signal, dtype=float)
        response_state, friction_e = organ.sense_and_interact(action)

        # 자아 경계 (Markov Blanket) 형성 판단
        is_self_domain = friction_e < 0.1

        return {
            "organ": organ_name,
            "friction_error": friction_e,
            "is_self_controlled": is_self_domain,
            "response_preview": response_state[:4].tolist()
        }
