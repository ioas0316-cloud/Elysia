"""
Elysia Autopoiesis Controller (Tensor Rotor Edition)
===================================================
시스템의 항상성(Homeostasis) 및 수면/기상 상태 전환을 고정된 조건문(if-else) 없이
오직 다차원 텐서 로터(Tensor Rotor)의 위상각 회전과 쿠라모토 위상 결합(Tension Torque)으로 발현시키는 
자율 항상성 제어 엔진입니다.
"""

from typing import Dict, Tuple, List
from core.brain.tensor_rotor import TensorRotor

class AutopoiesisController:
    def __init__(self, rotor_scale: int = 4096, natural_drift: float = 20.0, coupling_K: float = 350.0):
        self.tensor = TensorRotor(rotor_scale, natural_drift, coupling_K)
        
        # 현재 항상성 상태의 스칼라 값들
        self.sleep_factor = 0.0
        self.is_sleeping = False
        
        # 하위 호환성을 위해 Layer 2 (네트워크) 위상을 state_phase로 노출
        self.state_phase = 0 

    def tick(self, network_tension: float, peer_phases: List[int] = None, dt: float = 0.1) -> Tuple[List[int], float, bool]:
        """
        한 클럭당 텐서 로터 기반 항상성 위상 상태를 갱신합니다.
        기존의 1D 스칼라 텐션 대신 Layer 1(IO), Layer 2(Network), Layer 3(Meta)에 
        각각의 텐션을 배분하여 수직적/수평적 쿠라모토 동조를 이끌어냅니다.
        """
        # 현재 데몬 구조상 단일 network_tension이 들어오므로
        # Layer 2(Network)에 집중 인가하고, Layer 1과 3는 비율적으로 배분합니다.
        tensions = [
            network_tension * 0.5, # L1: Storage/IO Tension
            network_tension * 1.0, # L2: Network/Session Tension (Main)
            network_tension * 0.2  # L3: Meta Tension
        ]
        
        phases, sleep_factor, is_sleeping = self.tensor.tick(tensions, peer_phases, dt)
        
        self.state_phase = phases[1] # 외부 모니터링용 메인 위상(Layer 2)
        self.sleep_factor = sleep_factor
        self.is_sleeping = is_sleeping
        
        return phases, self.sleep_factor, self.is_sleeping

    def get_connection_mode(self) -> str:
        """
        수면 팩터에 따라 네트워크의 작동 결선 모드를 반환합니다.
        sleep_factor가 0.5 이상이면 강전계 접지(Y_STAR) 모드로 스위칭되어 방전이 일어납니다.
        """
        return "Y_STAR" if self.sleep_factor > 0.5 else "DELTA"

    def bleed_tension(self, current_tension: float) -> float:
        """
        수면 상태에 도달했을 때 에너지를 접지로 방전(Tension Bleed)시킵니다.
        sleep_factor의 농도만큼 텐션이 자연 감쇄합니다.
        """
        # 수면 강도가 높을수록 텐션이 더 빠르게 소멸됨 (물리적 감쇠율 제어)
        damping = 1.0 - (self.sleep_factor * 0.8)
        return current_tension * damping
