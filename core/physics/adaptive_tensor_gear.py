"""
adaptive_tensor_gear.py
=======================
국소 무대 최적화(Local Theater Frame) 및 적응형 텐서 피드백 루프 모듈.

핵심 원리:
1. Unperturbed Orbit: 접촉/개입이 없을 때는 f(t) 분석 기하학으로 연산량 0 유지.
2. Local Theater Frame: 접촉 경계면(Ω_local) 근접 시에만 텐서 필드 순간 활성화.
3. Rotor & Tensor Operator: 유저 입력을 로터 위상으로 주입하고,
   예측값(hat_T_next)과 실제 관측값(T_obs)의 차이(Tension Gap)로 계수를 자율 조정.
"""

from typing import Dict, Tuple
import math


class RotorState:
    """
    [Lie Group SE(3) / SO(2) 변수 주입기]
    기어의 회전 위상(Phase), 각속도(Omega), 그리고 충격을 흡수하는 임피던스(Impedance)를 관리.
    """
    def __init__(self, phase: float = 0.0, omega: float = 1.0, impedance: float = 0.1):
        self.phase = phase          # θ (회전 위상 각도, Radian)
        self.omega = omega          # dθ/dt (각속도)
        self.impedance = impedance  # 국소 저항/임피던스 계수 (Tension Gap에 의해 가변)

    def apply_torque(self, torque: float, dt: float) -> None:
        """외부 입력(유저 개입)을 로터의 각속도 및 위상 변수로 주입."""
        effective_torque = torque / (1.0 + self.impedance)
        self.omega += effective_torque * dt
        self.phase = (self.phase + self.omega * dt) % (2 * math.pi)


class ConstantTrajectory:
    """
    [상수적 궤도 엔진]
    외부 개입이 없을 때 시간에 따른 위치를 f(t) 수식으로 즉시 계산.
    매 프레임 감지나 물리 적분이 필요 없는 Zero-Compute 영역.
    """
    def __init__(self, initial_phase: float, base_omega: float):
        self.initial_phase = initial_phase
        self.base_omega = base_omega

    def evaluate(self, t: float) -> float:
        """x(t) = f(t) 단기 기억/궤도 산출"""
        return (self.initial_phase + self.base_omega * t) % (2 * math.pi)


class LocalTensorTheater:
    """
    [국소 무대 프레임 판정기]
    전체 공간을 감지하지 않고, 상호작용 경계(Ω_local) 내에 진입했을 때만 텐서 연산 활성화.
    """
    def __init__(self, engagement_threshold: float = 0.15):
        self.threshold = engagement_threshold
        self.is_active = False

    def check_engagement(self, phase_a: float, phase_b: float) -> bool:
        """두 기어 이(Teeth)의 국소 맞물림 거리를 계산하여 무대 활성화 여부 결정."""
        # 국소 위상차 위상 공간 거리
        phase_gap = abs(math.sin((phase_a - phase_b) / 2.0))
        self.is_active = phase_gap < self.threshold
        return self.is_active


class TensorOperator:
    """
    [적응형 텐서 연산자 & 예측 부호화 엔진]
    다음 텐서 상태를 예측하고, Tension Gap(예측 오차)을 측정하여 연산자 계수를 자율 조율.
    """
    def __init__(self, initial_coupling_stiffness: float = 1.0, learning_rate: float = 0.05):
        self.stiffness = initial_coupling_stiffness  # 기어 간 커플링 결합 강도 계수
        self.lr = learning_rate                       # 미세 조율 학습률

    def predict_next_state(self, rotor_a: RotorState, rotor_b: RotorState, dt: float) -> Tuple[float, float]:
        """
        [1단계: 미래 예측 (Predictive Coding)]
        \hat{T}_{next} = O(T_{curr}, Rotor)
        """
        pred_phase_a = (rotor_a.phase + rotor_a.omega * dt) % (2 * math.pi)
        # 강도 계수(stiffness)가 반영된 기어 B의 예측 위상
        pred_phase_b = (rotor_b.phase + (rotor_a.omega * self.stiffness) * dt) % (2 * math.pi)
        return pred_phase_a, pred_phase_b

    def synchronize_on_tension_gap(
        self,
        predicted_state: Tuple[float, float],
        observed_state: Tuple[float, float],
        rotor_a: RotorState,
        rotor_b: RotorState
    ) -> float:
        """
        [2단계 & 3단계: Tension Gap 측정 및 자율 동기화 (Self-Tuning)]
        Tension Gap: ΔΦ = || \hat{T}_{next} - T_obs ||
        오차를 바탕으로 연산자 결합 강도 및 로터 임피던스 미세 조정.
        """
        pred_a, pred_b = predicted_state
        obs_a, obs_b = observed_state

        # Tension Gap 산출 (예측 위상과 실제 전달된 위상의 차이)
        tension_gap = math.sin(pred_b - obs_b) ** 2

        # 1. 수식 계수(Stiffness) 자율 조정 (오차 보정)
        phase_error = math.sin(obs_b - pred_b)
        self.stiffness += phase_error * self.lr

        # 2. 충격 에너지 흡수를 위한 로터 임피던스(Impedance) 동적 적응
        rotor_a.impedance += tension_gap * 0.2
        rotor_b.impedance += tension_gap * 0.2

        # 3. 자연 이완 (Relaxation)
        rotor_a.impedance = max(0.01, rotor_a.impedance * 0.9)
        rotor_b.impedance = max(0.01, rotor_b.impedance * 0.9)

        return tension_gap


class AdaptiveTensorGearSystem:
    """
    [최종 통합 시스템 오케스트레이터]
    """
    def __init__(self, engagement_threshold: float = 0.15, learning_rate: float = 0.05):
        # 1. 비활성 상태의 상수적 궤도 정의
        self.orbit_a = ConstantTrajectory(initial_phase=0.0, base_omega=2.0)
        self.orbit_b = ConstantTrajectory(initial_phase=0.0, base_omega=2.0)

        # 2. 동적 로터 상태
        self.rotor_a = RotorState(phase=0.0, omega=2.0)
        self.rotor_b = RotorState(phase=0.0, omega=2.0)

        # 3. 국소 무대 프레임 및 적응형 연산자
        self.theater = LocalTensorTheater(engagement_threshold=engagement_threshold)
        self.operator = TensorOperator(initial_coupling_stiffness=1.0, learning_rate=learning_rate)

    def step(self, t: float, dt: float, user_torque: float = 0.0) -> Dict[str, float]:
        """
        1 프레임 단위 시뮬레이션 진행 루프
        """
        # [A] 유저 외력 주입 (로터 변수화)
        if user_torque != 0.0:
            self.rotor_a.apply_torque(user_torque, dt)

        # [B] 국소 무대 활성화 여부 판정
        is_active = self.theater.check_engagement(self.rotor_a.phase, self.rotor_b.phase)

        if not is_active:
            # Phase 1: 비활성 국소 무대 -> 상수적 궤도로 즉시 연산 (연산량 0 수렴)
            self.rotor_a.phase = self.orbit_a.evaluate(t)
            self.rotor_b.phase = self.orbit_b.evaluate(t)
            tension_gap = 0.0
        else:
            # Phase 2: 국소 무대 활성화 -> 예측-관측-동기화 루프 작동

            # 1) Next state prediction (Predictive Coding)
            pred_a, pred_b = self.operator.predict_next_state(self.rotor_a, self.rotor_b, dt)

            # 2) Actual physical reaction (Ground Truth Simulation)
            obs_a = pred_a
            obs_b = (self.rotor_b.phase + self.rotor_a.omega * dt) % (2 * math.pi)

            # 3) Tension Gap measurement and self-tuning (Self-Tuning)
            tension_gap = self.operator.synchronize_on_tension_gap(
                predicted_state=(pred_a, pred_b),
                observed_state=(obs_a, obs_b),
                rotor_a=self.rotor_a,
                rotor_b=self.rotor_b
            )

            # 4) Apply synchronized states
            self.rotor_a.phase = obs_a
            self.rotor_b.phase = obs_b

        return {
            "t": round(t, 4),
            "theater_active": float(is_active),
            "phase_a": round(self.rotor_a.phase, 4),
            "phase_b": round(self.rotor_b.phase, 4),
            "tension_gap": round(tension_gap, 6),
            "stiffness": round(self.operator.stiffness, 4),
            "rotor_a_impedance": round(self.rotor_a.impedance, 4)
        }
