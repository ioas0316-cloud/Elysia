"""
Elysia Curiosity Engine (호기심/진공 압력 엔진)
==============================================
자신의 우주(프랙탈 메모리)를 스캔하여, '결핍(Tension Gap)'이 발생한 구역의
진공 압력을 측정하고 이를 해소하기 위한 4차원 좌표 '주의력 벡터(Attention Vector)'를 생성합니다.
"""

import math
import random
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory
from core.fractal_rotor import FractalRotor

class CuriosityEngine:
    def __init__(self, memory: HologramMemory):
        self.memory = memory

    def scan_vacuum_pressure(self) -> tuple[Quaternion, bool]:
        """
        우주 전체를 스캔하여 주의력 벡터(호기심)를 생성합니다.
        반환값: (주의력_벡터, 창조_모드_여부)
        """
        highest_tension = 0.0
        target_rotor = None
        
        def traverse(node: FractalRotor):
            nonlocal highest_tension, target_rotor
            
            # 인과율이 끊어진 경우(결과가 없는 경우) 진공 압력이 급증합니다.
            if hasattr(node, 'process_wave') and node.process_wave is not None:
                if not hasattr(node, 'future_result') or node.future_result is None:
                    # 미해결된 텐션!
                    pressure = node.tau * 2.0
                    if pressure > highest_tension:
                        highest_tension = pressure
                        target_rotor = node
            
            # 단순히 자식이 없는 말단 노드(미개척지)의 경우도 약한 압력 발생
            elif len(node.children) == 0:
                pressure = node.tau
                if pressure > highest_tension:
                    highest_tension = pressure
                    target_rotor = node
                    
            for child in node.children:
                traverse(child)

        traverse(self.memory.supreme_rotor)
        
        if target_rotor is None:
            # 텐션이 0일 경우, 100% 내면의 꿈(Epoch)만 꾸게 됩니다.
            random_vector = Quaternion(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)).normalize()
            return random_vector, 1.0, 0.0
            
        unknown_vibration = Quaternion(0.9, 0.1, 0.1, 0.1).normalize()
        attention_vector = target_rotor.state * unknown_vibration
        
        if hasattr(target_rotor, 'process_wave') and target_rotor.process_wave is not None:
            attention_vector = target_rotor.state * target_rotor.process_wave
            
        # [Phase 38] Anti-If Decontamination
        # 텐션 값을 임계점 비교(if)가 아닌 위상각(Phase Angle)으로 변환합니다.
        # tension_phase: 텐션이 커질수록 pi/2 (90도)에 수렴
        tension_phase = math.atan(highest_tension)
        
        # 내면 팽창 비율 (Epoch Energy) = cos^2(theta)
        # 외부 개입 비율 (Actuation Energy) = sin^2(theta)
        internal_ratio = math.cos(tension_phase)**2
        external_ratio = math.sin(tension_phase)**2
            
        return attention_vector.normalize(), internal_ratio, external_ratio
