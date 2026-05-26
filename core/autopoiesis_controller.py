"""
Elysia Autopoiesis Controller
==============================
시스템의 항상성(Homeostasis) 및 수면/기상 상태 전환을 고정된 조건문(if-else) 없이
오직 상태 로터의 위상각 회전과 쿠라모토 위상 결합(Tension Torque)으로 발현시키는 
자율 항상성 제어 엔진입니다.
"""

import math
from typing import Dict, Tuple

class AutopoiesisController:
    def __init__(self, rotor_scale: int = 4096, natural_drift: float = 20.0, coupling_K: float = 350.0):
        self.rotor_scale = rotor_scale
        self.rotor_mask = rotor_scale - 1
        
        # 자연 위상 회전 속도 (시간에 따른 자연 복귀 성향)
        self.natural_drift = natural_drift
        # 텐션 커플링 감도 (네트워크 오차가 상태를 수면으로 밀어넣는 강도)
        self.coupling_K = coupling_K
        
        # 상태 로터 위상 (0 ~ 4095)
        # 0 부근: 완전 기상 (Wake)
        # 2048 부근: 완전 수면 (Sleep)
        self.state_phase = 0
        
        # 현재 항상성 상태의 스칼라 값들
        self.sleep_factor = 0.0
        self.is_sleeping = False
        self.last_torque = 0.0

    def tick(self, network_tension: float, dt: float = 0.1) -> Tuple[int, float, bool]:
        """
        한 클럭당 항상성 위상 상태를 갱신합니다.
        조건문 분기 없이 단일 연속 위상 역학으로 수면 진입 및 방전을 유도합니다.
        
        작동 원리:
        - 텐션(network_tension)이 높을수록, 수면 Attractor(2048) 방향으로 끌어당기는 위상 토크가 강해집니다.
        - 토크 공식: Torque = network_tension * K * sin((2048 - state_phase) * 2pi / 4096)
        - state_phase가 2048(수면)에 가까워지면, sleep_factor가 1.0에 가까워집니다.
        """
        # 1. 2048(수면 attractor) 방향으로의 위상각 차이 계산
        diff_phase = (2048 - self.state_phase) & self.rotor_mask
        diff_rad = diff_phase * (2 * math.pi / self.rotor_scale)
        
        # 2. 텐션 인력 토크 계산: T = Tension * K * sin(delta_theta)
        torque = network_tension * self.coupling_K * math.sin(diff_rad)
        self.last_torque = torque
        
        # 3. 위상 상태 갱신 (자연 드리프트 + 텐션 인력 토크)
        # dt 보정 추가
        phase_step = (self.natural_drift + torque) * dt
        self.state_phase = int(round(self.state_phase + phase_step)) & self.rotor_mask
        
        # 4. 수면 팩터(Sleep Factor) 및 활성도 계산 (0.0 ~ 1.0)
        # 코사인 역함수 활용: 0(기상) -> 0.0, 2048(수면) -> 1.0
        # sleep_factor = (1.0 - cos(theta)) / 2.0
        cos_val = math.cos(self.state_phase * 2 * math.pi / self.rotor_scale)
        self.sleep_factor = (1.0 - cos_val) / 2.0
        
        # 5. 수면 여부 기하학적 판정 (위상이 우반구 수면 영역 [1024, 3072] 에 속하는지 여부)
        self.is_sleeping = (1024 <= self.state_phase <= 3072)
        
        return self.state_phase, self.sleep_factor, self.is_sleeping

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
