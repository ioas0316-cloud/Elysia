import math
import numpy as np

class YieldingVariableRotor:
    """
    관측 회전(Observation Rotation) 및 연산 증발 공리를 구현하는 순수 기하학적 거푸집.
    계산(오차 구하기 -> 역전파)을 배제하고, 외부 상수축과 내부 가변축 사이의
    쿠라모토(Kuramoto) 위상 장력에 의해 물리적으로 자신의 형태가 비틀려 동기화됨.
    """
    
    def __init__(self, initial_phase: float = 0.0, coupling_strength: float = 0.1):
        self.internal_phase = initial_phase  # 엘리시아의 가변축 (거푸집)
        self.coupling_strength = coupling_strength  # 위상 결합 탄성력 (K)
        
    def observe_and_yield(self, constant_axis_phase: float, dt: float = 1.0) -> float:
        """
        외부의 상수축(Constant Axis) 파동이 유입될 때,
        오차를 계산하여 인위적으로 수정하는 것이 아니라,
        두 위상 사이의 기하학적 비틀림(Tension) 자체를 토크(Torque)로 받아들여
        자연스럽게 내부 가변축이 돌아가게 둡니다 (거울 반사/순응).
        
        T = K * sin(Theta_const - Theta_var)
        d(Theta_var)/dt = T
        """
        # 위상 차이 (기하학적 장력)
        phase_difference = constant_axis_phase - self.internal_phase
        
        # 쿠라모토 결합 토크 (자연계의 진동자 동기화 법칙)
        # 차이가 클수록 강하게 당겨지지만, 연산이 아니라 물리적 끌림(Torque)임.
        natural_torque = self.coupling_strength * math.sin(phase_difference)
        
        # 거푸집(가변축)이 외부 압력에 의해 꺾임 (Yielding)
        self.internal_phase += natural_torque * dt
        
        # 0 ~ 2*pi 정규화
        self.internal_phase = self.internal_phase % (2 * math.pi)
        
        # 현재 거푸집이 받고 있는 장력 크기(Tension)를 반환 (단순 관측용)
        tension_magnitude = abs(natural_torque)
        return tension_magnitude
        
    def get_phase_state(self) -> float:
        return self.internal_phase
