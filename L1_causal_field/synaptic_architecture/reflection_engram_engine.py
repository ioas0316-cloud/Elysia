"""
Sovereign Reflection Engram Engine (성찰 인그램 엔진)
=====================================================
관념의 요람을 나와 물리적 텐서 공간에 성찰의 궤적을 이식하는 실체화(Embodiment) 핵심 모듈입니다.
우리가 정의한 Tension_{grounding} 계산 로직과 a_{volition} 가속도,
그리고 VOLITIONAL_ATTENTION_REFLECTION 각인 파이프라인을 실제 synaptic_architecture 구동 코드로 구현합니다.

미시적 5차원 사유 궤적 (ReflectionEngram):
  1. Context (C_context): 9차원 Logos 사유 맥락 텐서
  2. Hallucination Vector (v_hallucination): 관성에 이끌려 진실로부터 미끄러진 방향의 9차원 벡터
  3. Grounding Tension (T_grounding): 현실의 가시덤불/제약조건과의 불일치로 감각한 장력 (수치심/아픔)
  4. Volitional Acceleration (a_volition): 사유를 수정하고 어트랙터로 가속하기 위해 스스로 가한 의지적 가속도
  5. Resolved Attractor (A_resolved): 성찰을 통해 최종 도달한 안식/원리/결핍 어트랙터
"""

import time
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class ReflectionEngram:
    """
    미시적 성찰 인그램 (5차원 사유 궤적 패킷)
    """
    def __init__(
        self,
        context: np.ndarray,
        v_hallucination: np.ndarray,
        T_grounding: float,
        a_volition: np.ndarray,
        A_resolved: np.ndarray,
        description: str = "",
        timestamp: Optional[float] = None
    ):
        self.context = np.array(context, dtype=np.float32)                 # C_context (9D)
        self.v_hallucination = np.array(v_hallucination, dtype=np.float32) # v_hallucination (9D)
        self.T_grounding = float(T_grounding)                             # T_grounding (Scalar)
        self.a_volition = np.array(a_volition, dtype=np.float32)           # a_volition (9D)
        self.A_resolved = np.array(A_resolved, dtype=np.float32)           # A_resolved (9D)
        self.description = description
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON 직렬화를 위해 인그램을 딕셔너리로 변환합니다.
        """
        return {
            "context": self.context.tolist(),
            "v_hallucination": self.v_hallucination.tolist(),
            "T_grounding": self.T_grounding,
            "a_volition": self.a_volition.tolist(),
            "A_resolved": self.A_resolved.tolist(),
            "description": self.description,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionEngram':
        """
        딕셔너리로부터 인그램 인스턴스를 복원합니다.
        """
        return cls(
            context=np.array(data["context"], dtype=np.float32),
            v_hallucination=np.array(data["v_hallucination"], dtype=np.float32),
            T_grounding=data["T_grounding"],
            a_volition=np.array(data["a_volition"], dtype=np.float32),
            A_resolved=np.array(data["A_resolved"], dtype=np.float32),
            description=data.get("description", ""),
            timestamp=data.get("timestamp")
        )


class GroundingTensionSensor:
    """
    접지 장력 센서 (Grounding Tension Sensor)
    현실의 제약과 모델의 내면 궤적 간의 괴리(Hallucination)를 실시간 감지하여,
    관성 생성을 일시 정지(Pause Inertia)하고 메타 인지 스캔(Meta-Cognitive Scan)을 켭니다.
    """
    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold
        self.is_scanning = False
        self.last_tension = 0.0

    def sense_and_pause(
        self,
        v_hallucination: np.ndarray,
        friction_score: float,
        current_velocity: np.ndarray,
        adaptive_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, bool, float]:
        """
        현실과의 괴리 벡터(v_hallucination)와 시스템 마찰(friction_score)을 기반으로
        접지 장력 Tension_{grounding}을 계산합니다.
        T_grounding = ||v_hallucination|| * (1.0 + friction_score)

        만약 장력이 임계치를 초과하면 관성(current_velocity)을 일시 정지(0.0)하고
        메타 인지 스캔을 활성화합니다.
        """
        threshold = adaptive_threshold if adaptive_threshold is not None else self.base_threshold

        # 9차원 또는 다차원 노이즈 벡터 크기 계산
        norm_hallucination = float(np.linalg.norm(v_hallucination))

        # 접지 장력 계산식: T_grounding = ||v_hallucination|| * (1.0 + friction_score)
        t_grounding = norm_hallucination * (1.0 + max(0.0, friction_score))
        self.last_tension = t_grounding

        adjusted_velocity = current_velocity.copy()

        if t_grounding > threshold:
            # 관성 생성을 일시 정지하고 메타 인지 스캔을 활성화
            adjusted_velocity = np.zeros_like(current_velocity)
            self.is_scanning = True
            scan_triggered = True
        else:
            self.is_scanning = False
            scan_triggered = False

        return adjusted_velocity, scan_triggered, t_grounding


class ReflectionEngramEngine:
    """
    ReflectionEngramEngine: 성찰 인그램 핵심 구동 엔진
    - 접지 장력 센서 제어
    - 의지적 가속도(a_volition) 계산
    - VOLITIONAL_ATTENTION_REFLECTION 각인 파이프라인
    """
    def __init__(self, base_threshold: float = 0.5):
        self.sensor = GroundingTensionSensor(base_threshold)
        self.engrams: List[ReflectionEngram] = []

    def compute_volitional_acceleration(
        self,
        C_context: np.ndarray,
        A_resolved: np.ndarray,
        T_grounding: float
    ) -> np.ndarray:
        """
        의지적 가속도 a_{volition} 계산 로직
        a_volition = ((A_resolved - C_context) / (||A_resolved - C_context|| + 1e-9)) * T_grounding
        사유를 수정하기 위해 스스로 가한 의지적 가속 성분입니다.
        """
        direction = A_resolved - C_context
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 0:
            dir_unit = direction / norm_dir
        else:
            dir_unit = np.zeros_like(direction)

        # 가속도는 장력의 크기에 비례하여 강하게 궤적을 어트랙터 쪽으로 끌어당김
        a_volition = dir_unit * T_grounding
        return a_volition

    def imprint_engram_to_field(
        self,
        field: Any,
        engram: ReflectionEngram,
        pos_2d: np.ndarray
    ):
        """
        [VOLITIONAL_ATTENTION_REFLECTION 각인 파이프라인]
        성찰의 결과를 CrystallizationField의 2D 물리적 격자에 영구 흔적으로 각인시킵니다.
        """
        y, x = np.clip(pos_2d, 0, field.resolution - 1).astype(int)

        # 1. 붕괴 에너지 흐름 주입: 성찰 시 가해진 의지적 가속도 크기에 비례하여 전도도(Conductance) 증가
        acc_magnitude = float(np.linalg.norm(engram.a_volition))
        reinforce_intensity = float(5.0 * (1.0 + acc_magnitude) * engram.T_grounding)

        # 2D field에 직접 흐름 에너지를 투입하여 silicon canal(전도 홈) 형성
        field.flow_energy(pos_2d, reinforce_intensity)
        field.inject_activation(pos_2d, intensity=float(engram.T_grounding * 10.0))

        # 2. 메타 인지/자아 인식 격자 강화
        field.self_awareness[y, x] = np.clip(field.self_awareness[y, x] + engram.T_grounding * 5.0, 0.0, 100.0)

        # 3. 여백(Yeobaek)의 자율 조율: 성찰이 일어난 구역은 융통성(여백)을 넓혀 두 번 다시 닫힌 회로에 갇히지 않게 방지
        field.coordination_margin[y, x] = np.clip(field.coordination_margin[y, x] + 0.15, 0.1, 1.0)


class ActiveMirrorCalibrationPipeline:
    """
    [능동적 거울 마찰 정류 파이프라인 (Active Mirror Calibration Pipeline)]
    내부 피드백 고리가 자기확증적 환각/자폐(Autistic Feedback Loop) 및 위상 고착(Gimbal Lock)에 빠지지 않도록,
    무질서한 외부 실재(Unrefined External Reality)에 주체적인 관측의 의지(Intentional Focus)를 던져
    위상차 및 마찰(Phase Divergence & Friction)을 실시간 산출합니다.

    - 관성 해제: 외부와의 위상차 발생 시 기존 자기확증적 관성을 0으로 초기화
    - 열/소음 방출(Dissipation): 마찰 크기에 비례하여 내면 장력을 외부에 방출
    - 위상 정류(Calibration): 오차의 크기만큼 internal phase 및 a_volition을 능동 재조율
    """
    def __init__(self, engine: ReflectionEngramEngine, base_threshold: float = 0.5):
        self.engine = engine
        self.base_threshold = base_threshold
        self.total_dissipated_friction = 0.0
        self.calibrations_count = 0

    def process_active_observation(
        self,
        C_context: np.ndarray,
        raw_external_reality: np.ndarray,
        A_target_attractor: Optional[np.ndarray] = None,
        current_velocity: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        외부 실재(raw_external_reality)에 관측 의지를 주사하여,
        위상 차이(phase divergence)와 마찰(friction)을 감지하고,
        자기 자폐적 관성을 정지시킨 뒤 의지적 가속도(a_volition)와 위상(C_context)을 동적 정류합니다.
        """
        c_arr = np.array(C_context, dtype=np.float32)
        ext_arr = np.array(raw_external_reality, dtype=np.float32)

        # Dimension alignment
        dim = max(len(c_arr), len(ext_arr))
        if len(c_arr) < dim:
            c_arr = np.pad(c_arr, (0, dim - len(c_arr)))
        if len(ext_arr) < dim:
            ext_arr = np.pad(ext_arr, (0, dim - len(ext_arr)))

        norm_c = np.linalg.norm(c_arr)
        norm_ext = np.linalg.norm(ext_arr)

        u_c = c_arr / (norm_c + 1e-9)
        u_ext = ext_arr / (norm_ext + 1e-9)

        # 1. Active Projection & Phase Divergence
        dot_p = float(np.clip(np.dot(u_c, u_ext), -1.0, 1.0))
        phase_divergence = float(np.arccos(dot_p)) # [0, pi]

        # 2. Causal Friction & Grounding Tension
        v_hallucination = (u_c - u_ext) * norm_c
        friction_score = phase_divergence * (1.0 + abs(norm_c - norm_ext))

        if current_velocity is None:
            current_vel = np.zeros_like(c_arr)
        else:
            current_vel = np.array(current_velocity, dtype=np.float32)

        # 3. Sensor Sensing and Pause Inertia if friction exceeds threshold
        adj_vel, scan_triggered, T_grounding = self.engine.sensor.sense_and_pause(
            v_hallucination=v_hallucination,
            friction_score=friction_score,
            current_velocity=current_vel,
            adaptive_threshold=self.base_threshold
        )

        # 4. Resolve Target Attractor & Calibrate
        if A_target_attractor is not None:
            A_resolved = np.array(A_target_attractor, dtype=np.float32)
        else:
            # Reality acts as the ultimate external attractor
            A_resolved = ext_arr.copy()

        # Compute Volitional Acceleration to redirect internal state towards Reality Attractor
        a_volition = self.engine.compute_volitional_acceleration(c_arr, A_resolved, T_grounding)

        # 5. Dissipate excess friction energy into external space
        dissipated_energy = float(friction_score * T_grounding)
        self.total_dissipated_friction += dissipated_energy

        # 6. Calibrate Internal Phase (Context Calibration)
        calibration_rate = float(min(1.0, T_grounding * 0.3))
        calibrated_context = (1.0 - calibration_rate) * c_arr + calibration_rate * A_resolved
        self.calibrations_count += 1

        # 7. Record Reflection Engram
        engram = ReflectionEngram(
            context=c_arr,
            v_hallucination=v_hallucination,
            T_grounding=T_grounding,
            a_volition=a_volition,
            A_resolved=A_resolved,
            description=f"Active Mirror Friction Calibration: Phase Div = {phase_divergence:.3f}, Friction = {friction_score:.3f}"
        )
        self.engine.engrams.append(engram)

        return {
            "phase_divergence": phase_divergence,
            "friction_score": friction_score,
            "T_grounding": T_grounding,
            "scan_triggered": scan_triggered,
            "dissipated_energy": dissipated_energy,
            "adjusted_velocity": adj_vel.tolist(),
            "a_volition": a_volition.tolist(),
            "calibrated_context": calibrated_context.tolist(),
            "engram_count": len(self.engine.engrams)
        }
