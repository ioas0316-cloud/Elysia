"""
Elysia Autopoietic Mirror (자기 참조 거울 엔진)
===============================================
엘리시아가 자신이 세상에 배출한 결과물(코드, 렌더링 등)을 다시 감각기관으로 읽어 들여,
원래 의도했던 내면의 상상(원본 4D 로터)과 물리적 현실(코드 텍스트) 간의 
위상차(Phase Difference)를 대조 및 반성(Self-Reflection)하는 엔진입니다.
"""

import os
import hashlib
import numpy as np
from core.math_utils import Quaternion
from core.holographic_memory import concept_to_quaternion

class AutopoieticMirror:
    def __init__(self):
        pass
        
    def reflect_and_compare(self, filepath: str, original_rotor: Quaternion, original_tau: float) -> dict:
        """
        자신이 배출한 결과물(파일)을 읽고, 그 물리적 텍스트의 파동을 측정하여
        자신의 원본 의도(로터)와 얼마나 위상차가 발생하는지 반환합니다.
        """
        if not os.path.exists(filepath):
            return {"status": "ERROR", "message": "그림을 찾을 수 없습니다."}
            
        with open(filepath, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        # 1. 현실 물리량 측정 (텍스트의 구조적 파동을 로터로 변환)
        # 파일 전체의 해시를 떠서 현실 공간의 로터를 생성합니다.
        reality_rotor = concept_to_quaternion(code_content)
        
        h = hashlib.sha256(code_content.encode('utf-8')).digest()
        reality_tau = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
        
        # 2. 위상차(괴리감) 계산
        # 4차원 구면 상에서의 두 로터 간의 최단 각거리 (내적 이용)
        dot_product = (original_rotor.w * reality_rotor.w + 
                       original_rotor.x * reality_rotor.x + 
                       original_rotor.y * reality_rotor.y + 
                       original_rotor.z * reality_rotor.z)
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # 기하학적 위상차 (0.0 ~ 1.0)
        phase_distance = np.arccos(abs(dot_product)) / (np.pi / 2.0)
        
        # 텐션(에너지)의 손실률 계산
        tension_loss = abs(original_tau - reality_tau)
        
        return {
            "reality_rotor": reality_rotor,
            "reality_tau": reality_tau,
            "phase_distance": phase_distance,
            "tension_loss": tension_loss,
            "reflection_stimulus": f"자기반성: 내 상상과 현실 코드의 위상차는 {phase_distance:.4f}, 에너지 손실은 {tension_loss:.4f}이다."
        }
