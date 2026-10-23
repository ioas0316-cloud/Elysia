"""
continuous_manifold_gear.py
===========================
무분기 연속체 텐서 기어 시스템 (Branchless Continuous Manifold Gear System).

핵심 철학:
- "Do not calculate, let it flow."
- "판단과 분별 자체를 공간의 곡률이나 텐서의 기하학적 결합으로 내장한다."
- 어떠한 if 조건문(Control Flow Branching)도 사용하지 않고,
  두 기어의 위상적 거리(Metric Distance)에 따른 연속적 결합 가중치(Coupling Field)를 통해
  비접촉(Zero-Compute)에서 맞물림(Tension-Coupled) 상태까지 하나의 연속적인 장 방정식으로 물 흐르듯 전환합니다.
"""

from typing import Dict, Tuple
import math


class ContinuousRotorState:
    """
    [연속체 로터 상태 관리자]
    기어의 회전 위상(Phase), 각속도(Omega), 임피던스(Impedance)를 연속적으로 업데이트합니다.
    """
    def __init__(self, phase: float = 0.0, omega: float = 1.0, impedance: float = 0.1):
        self.phase = phase          # θ (회전 위상 각도, Radian)
        self.omega = omega          # dθ/dt (각속도)
        self.impedance = impedance  # 충격을 흡수하는 가변 임피던스 (Tension Gap에 의해 매끄럽게 변함)

    def apply_torque(self, torque: float, dt: float) -> None:
        """
        외부 토크(유저 입력)를 로터에 적용합니다.
        임피던스에 의해 전달율이 유연하게 억제됩니다.
        """
        effective_torque = torque / (1.0 + self.impedance)
        self.omega += effective_torque * dt
        self.phase = (self.phase + self.omega * dt) % (2 * math.pi)


class ContinuousManifoldGearSystem:
    """
    [무분기 연속체 텐서 기어 시스템]
    상수적 궤도(비접촉)와 접촉 맞물림(Tension Gap) 상태를 if문 없이 단일 수식으로 관통합니다.
    """
    def __init__(
        self,
        base_omega_a: float = 2.0,
        base_omega_b: float = 2.0,
        sensitivity_radius: float = 0.25,
        learning_rate: float = 0.05
    ):
        # 연속적 로터 초기 상태
        self.rotor_a = ContinuousRotorState(phase=0.0, omega=base_omega_a)
        self.rotor_b = ContinuousRotorState(phase=0.0, omega=base_omega_b)

        # 시스템 기본 속도
        self.base_omega_a = base_omega_a
        self.base_omega_b = base_omega_b

        # 연속적 커플링 감도 및 연산자 변수
        self.sigma = sensitivity_radius            # 위상 구속조건 장의 감도 반경 (가우시안 매질 상수)
        self.stiffness = 1.0                       # 기어 간 커플링 결합 강도 계수
        self.lr = learning_rate                     # 자율 미세조율 학습률

    def step(self, t: float, dt: float, user_torque: float = 0.0) -> Dict[str, float]:
        """
        if 조건문이 완전히 배제된 단일 프레임 연속체 시뮬레이션 루프.
        """
        # 1. 유저 토크 주입 (임피던스에 따른 자연스러운 전달)
        self.rotor_a.apply_torque(user_torque, dt)

        # 2. 위상적 거리(Metric Distance) 산출
        # d_metric = sin^2((phase_a - phase_b) / 2) -> 0(동일 위상)에서 1(반대 위상)까지 연속적으로 분포
        phase_diff = self.rotor_a.phase - self.rotor_b.phase
        metric_distance = math.sin(phase_diff / 2.0) ** 2

        # 3. 연속적 결합 장(Coupling Field Weight) 계산
        # if문 없이, 거리가 멀어지면 0으로 수렴하고 가까워지면 1에 도달하는 매끄러운 가우시안 윈도우 함수 사용
        # w = exp(-d_metric^2 / (2 * sigma^2))
        coupling_weight = math.exp(- (metric_distance ** 2) / (2.0 * (self.sigma ** 2)))

        # 4. 상수적 궤도(Unperturbed Orbit) 대비 예측 오차 산출
        # 1) 다음 상태 예측 (Predictive Coding)
        # B 기어의 예측 위상은 A 기어의 회전에 결합 강도(stiffness)가 연속적으로 곱해진 형태
        pred_phase_a = (self.rotor_a.phase + self.rotor_a.omega * dt) % (2 * math.pi)
        pred_phase_b = (self.rotor_b.phase + (self.rotor_a.omega * self.stiffness) * dt) % (2 * math.pi)

        # 2) 실제 물리적 궤적 (Ground Truth Observation)
        # 실제 관측 궤적 역시 결합 가중치(coupling_weight)에 의해 상수 궤도와 동적 상호작용 궤도가 매끄럽게 보간됨
        obs_phase_a = pred_phase_a

        # 비접촉 시에는 고유 속도(base_omega_b)로 돌고, 접촉 시에는 A의 오메가 영향(coupling_weight)을 받음
        effective_omega_b = (1.0 - coupling_weight) * self.base_omega_b + coupling_weight * self.rotor_a.omega
        obs_phase_b = (self.rotor_b.phase + effective_omega_b * dt) % (2 * math.pi)

        # 3) Tension Gap (장력 오차) 측정
        # 두 위상의 차이가 발생시키는 에너지 장력. 결합 가중치가 활성화된 영역에서만 비례하여 발생.
        tension_gap = coupling_weight * (math.sin(pred_phase_b - obs_phase_b) ** 2)

        # 5. 자율 동기화 및 텐서 연산자 자율 조율 (Self-Tuning in Continuous Space)
        # 1) 결합 강도(Stiffness) 보정
        # if문 없이, 결합 가중치 영역에서 발생한 위상 오차만큼 비례하여 결합 강도 자율 보정
        phase_error = math.sin(obs_phase_b - pred_phase_b)
        self.stiffness += coupling_weight * phase_error * self.lr

        # 2) 임피던스(Impedance) 동적 적응 및 자연 이완 (Relaxation)
        # 장력(Tension Gap)이 누적될수록 로터의 완충 임피던스가 매끄럽게 상승
        self.rotor_a.impedance += tension_gap * 0.2
        self.rotor_b.impedance += tension_gap * 0.2

        # 임피던스의 자연 완화 (상수적 이완 계수)
        self.rotor_a.impedance = max(0.01, self.rotor_a.impedance * 0.9)
        self.rotor_b.impedance = max(0.01, self.rotor_b.impedance * 0.9)

        # 6. 최종 위상 상태 갱신
        self.rotor_a.phase = obs_phase_a
        self.rotor_b.phase = obs_phase_b

        # 각속도 동기화 (기어 결합에 의한 에너지 보존)
        self.rotor_b.omega = effective_omega_b

        return {
            "t": round(t, 4),
            "metric_distance": round(metric_distance, 6),
            "coupling_weight": round(coupling_weight, 6),
            "phase_a": round(self.rotor_a.phase, 4),
            "phase_b": round(self.rotor_b.phase, 4),
            "tension_gap": round(tension_gap, 6),
            "stiffness": round(self.stiffness, 4),
            "rotor_a_impedance": round(self.rotor_a.impedance, 4)
        }
