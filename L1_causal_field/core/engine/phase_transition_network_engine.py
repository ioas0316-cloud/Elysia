"""
Elysia Core Engine: Phase Transition Network Engine (고정축-변수축 상전이 네트워크 동기화 엔진)
===================================================================================
3-Boundary Architecture (삼중 경계 시스템) 기반 차세대 네트워크 동기화 및 외란 완충 엔진.

1. 삼중 경계 아키텍처 (3-Boundary System):
   - 내부경계 (Invariant Anchor A_inv): 시스템의 본질적 고정축 (의도, 물리적 골격, 고유 ID, 포텐셜 우물 깊이 V_well).
   - 중간경계 (Middle Boundary / Markov Blanket Buffer): 유동적 중계 레이어.
     환경 압력(P_env) 및 위상 오차(q_err)를 흡수하며 ICE(고체) -> LIQUID(액체) -> GAS(기체) 상전이.
   - 외부경계 (Environmental Pressure P_env): 네트워크 지연(RTT), 패킷 손실, 지터 등 외부 노이즈 및 외란 파동.

2. 비판적 임계점 (Critical Threshold Discriminator):
   - 약한 충격 (Sub-critical Impact, q_err <= V_well):
     상전이 없이 탄성 울림(Elastic Vibration)으로 미세 충격을 완충하고 0으로 감쇄.
   - 강한 충격 (Super-critical Impact, q_err > V_well):
     구조 파괴 방지를 위해 결빙 해제.
     - 네트워크 지연/지터 유입 시: LIQUID(액체상) 상전이 -> 고정축 궤적 예측 연결 (프레임 멈춤/위치 튀김 방지).
     - 네트워크 단절/심각한 오차 시: GAS(기체상) 상전이 -> 엔트로피 분산 탐색 (크래시/데스윙 방지).

3. 음각 복구 및 거울 대칭 결빙 (Mirror Symmetry Re-ICE):
   - 재연결 또는 정밀 패킷 수신 시, 중간경계에 각인된 음각 흠집(위상 불일치 q_err)에 대한 거울 대칭 연산 적용:
     q_err -> 0으로 수렴시켜 안전하게 Re-ICE(결빙) 복원.

4. 통신 과부하 절감:
   - 결과 좌표/상태값을 매 프레임 송수신하는 대신, 고정축(A_inv)과 가변 다이얼 규칙만 공유하여 통신량 획기적 축소.
"""

import math
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class PhaseState(Enum):
    """중간경계의 상전이 상태 (Phase State)"""
    ICE = "ice"          # 고체상: 정시 도착, 완벽한 위상 고정 (Lattice Crystal)
    LIQUID = "liquid"    # 액체상: 지연 발생시 유동적 위상 궤적 예측 (Fluid Flow)
    GAS = "gas"          # 기체상: 단절/오류시 엔트로피 산란 및 범위 탐색 (Dispersed Gas)


class InvariantAnchor:
    """
    내부경계 (Invariant Anchor, A_inv):
    흔들리지 않는 시스템의 본질적 고정축.
    """
    def __init__(
        self,
        entity_id: str,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        intent_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        potential_well_depth: float = 0.5
    ):
        self.entity_id = entity_id
        self.base_position = np.array(base_position, dtype=np.float64)
        self.intent_vector = np.array(intent_vector, dtype=np.float64)
        self.mass = mass
        self.potential_well_depth = potential_well_depth  # V_well: 탄성 우물 깊이 임계치


class MiddleBoundaryBuffer:
    """
    중간경계 (Middle Boundary Buffer / Markov Blanket):
    내부경계와 외부경계 사이의 유동적 완충 및 상전이 해석 공간.
    """
    def __init__(self, anchor: InvariantAnchor):
        self.anchor = anchor
        self.current_phase: PhaseState = PhaseState.ICE
        self.phase_error: float = 0.0                      # q_err: 위상 오차 (음각 흠집)
        self.phase_angle: float = 0.0                      # \theta: 현재 위상 각도
        self.smoothed_position: np.ndarray = anchor.base_position.copy()
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self.elastic_damping: float = 0.85                  # 탄성 감쇄 계수
        self.liquid_viscosity: float = 0.92                 # 액체 유동 점성 계수
        self.concave_imprint: List[float] = []              # 음각 자국 백로그

    def evaluate_impact_and_transition(self, env_pressure: float, network_delay_ms: float, packet_loss_rate: float) -> PhaseState:
        """
        외부 환경 압력(P_env) 및 네트워크 상태를 받아 비판적 임계점(Critical Threshold) 판단 후 상전이
        """
        # 1. 외부 파동에 의한 위상 오차 q_err 산출
        delay_factor = network_delay_ms / 100.0
        calculated_q_err = env_pressure * (1.0 + delay_factor) + (packet_loss_rate * 2.0)
        self.phase_error = calculated_q_err
        self.concave_imprint.append(calculated_q_err)

        # 2. 비판적 임계점 (Critical Threshold) 판별
        v_well = self.anchor.potential_well_depth

        if calculated_q_err <= v_well:
            # [Sub-critical Impact]: 약한 충격 -> 탄성 복원 (ICE 유지 또는 감쇄)
            if self.current_phase == PhaseState.ICE:
                # ICE 상태 유지하며 탄성 진동 감쇄
                self.phase_error *= self.elastic_damping
            return self.current_phase

        # [Super-critical Impact]: 강한 충격 -> 상전이 비상 해제
        if packet_loss_rate > 0.4 or network_delay_ms > 300.0:
            # 네트워크 단절 또는 극심한 손실 -> GAS (기체상) 상전이로 엔트로피 분산
            self.current_phase = PhaseState.GAS
        else:
            # 네트워크 지연 / 지터 -> LIQUID (액체상) 상전이로 유동적 궤적 수용
            self.current_phase = PhaseState.LIQUID

        return self.current_phase

    def process_fluid_interpolation(self, dt: float) -> np.ndarray:
        """
        LIQUID 상전이 상태: 고정축(A_inv)의 의도를 바탕으로 클라이언트 내부 유동적 위상 궤적 연결.
        화면 멈춤(Freeze)이나 위치 튀김(Rubber-banding) 없이 매끄러운 수렴.
        """
        if self.current_phase == PhaseState.LIQUID:
            # 고정축 의도 방향으로 유동적 추진력 계산
            target_velocity = self.anchor.intent_vector / max(self.anchor.mass, 1e-5)
            self.velocity = self.velocity * self.liquid_viscosity + target_velocity * (1.0 - self.liquid_viscosity)
            self.smoothed_position += self.velocity * dt
            # 위상 오차 완화
            self.phase_error *= 0.95
        elif self.current_phase == PhaseState.GAS:
            # GAS 상태에서는 탐색 범위 확장을 위한 확률적 유체 운동
            noise = np.random.normal(0, 0.05, size=3)
            self.smoothed_position += (self.anchor.intent_vector * 0.5 + noise) * dt
        else:
            # ICE 상태: 고정축 지점에 완전 상전이 결빙
            self.smoothed_position = self.anchor.base_position.copy()

        return self.smoothed_position

    def apply_mirror_symmetry_re_ice(self) -> Dict[str, Any]:
        """
        음각 복구 및 거울 대칭 연산 (Mirror Symmetry Operation):
        중간경계에 축적된 위상 오차(q_err)에 거울 대칭 반전 행렬 M_mirror 적용.
        q_err -> 0으로 원자적 수렴시키고 ICE 상태로 재결빙.
        """
        initial_q_err = self.phase_error

        # 거울 대칭 대칭 연산: q_err_restored = q_err - M_mirror * q_err -> 0
        mirror_matrix = 1.0  # 반전 거울 축
        restored_q_err = initial_q_err - (mirror_matrix * initial_q_err)

        self.phase_error = restored_q_err
        self.current_phase = PhaseState.ICE
        self.anchor.base_position = self.smoothed_position.copy()
        self.concave_imprint.clear()

        return {
            "status": "re_ice_success",
            "initial_q_err": initial_q_err,
            "restored_q_err": restored_q_err,
            "final_phase": self.current_phase.value,
            "crystallized_position": self.anchor.base_position.tolist()
        }


class PhaseTransitionNetworkEngine:
    """
    고정축-변수축 상전이 네트워크 동기화 전체 컨트롤러 엔진.
    """
    def __init__(self, entity_id: str = "player_1"):
        self.anchor = InvariantAnchor(entity_id=entity_id, potential_well_depth=0.6)
        self.middle_boundary = MiddleBoundaryBuffer(self.anchor)

        # 통신량 비교 메트릭
        self.raw_packets_sent_bytes: int = 0
        self.phase_packets_sent_bytes: int = 0
        self.frame_freeze_count: int = 0
        self.crash_count: int = 0

    def update_server_state(self, intent: Tuple[float, float, float], base_pos: Tuple[float, float, float]):
        """서버 단에서 고정축 본질 정보(Invariant Anchor)갱신"""
        self.anchor.intent_vector = np.array(intent, dtype=np.float64)
        self.anchor.base_position = np.array(base_pos, dtype=np.float64)

    def simulate_network_frame(
        self,
        dt: float,
        env_pressure: float,
        rtt_ms: float,
        packet_loss_rate: float,
        packet_arrived: bool
    ) -> Dict[str, Any]:
        """
        1 프레임 네트워크 수신 및 상전이 처리 수행
        """
        # 통신 패킷 크기 계산
        # 기존 방식: 좌표(12B) + 속도(12B) + 상태값(16B) + 헤더(20B) = 60 bytes / frame
        self.raw_packets_sent_bytes += 60

        if packet_arrived:
            # 상전이 방식: 변수 다이얼 & 고정축 델타 수신 = 12 bytes / sync frame
            self.phase_packets_sent_bytes += 12

        # 1. 외부 환경 압력 반영 및 비판적 임계점 상전이 판단
        effective_loss = packet_loss_rate if not packet_arrived else packet_loss_rate * 0.2
        phase_state = self.middle_boundary.evaluate_impact_and_transition(
            env_pressure=env_pressure,
            network_delay_ms=rtt_ms,
            packet_loss_rate=effective_loss
        )

        # 2. 유동적 궤적 관측 및 보정
        current_pos = self.middle_boundary.process_fluid_interpolation(dt)

        # 3. 정밀 패킷 복구 수신 시 거울 대칭 Re-ICE 시도
        re_ice_result = None
        if packet_arrived and phase_state in (PhaseState.LIQUID, PhaseState.GAS):
            if self.middle_boundary.phase_error < self.anchor.potential_well_depth * 1.5:
                re_ice_result = self.middle_boundary.apply_mirror_symmetry_re_ice()

        # 기존 전통 방식의 freeze/crash 모의 비교
        traditional_freeze = not packet_arrived and rtt_ms > 150.0
        traditional_crash = packet_loss_rate > 0.5 and rtt_ms > 400.0

        if traditional_freeze:
            self.frame_freeze_count += 1
        if traditional_crash:
            self.crash_count += 1

        bandwidth_reduction_ratio = 1.0 - (
            self.phase_packets_sent_bytes / max(1, self.raw_packets_sent_bytes)
        )

        return {
            "dt": dt,
            "phase_state": self.middle_boundary.current_phase.value,
            "phase_error": float(self.middle_boundary.phase_error),
            "position": current_pos.tolist(),
            "re_ice_performed": re_ice_result is not None,
            "re_ice_details": re_ice_result,
            "bandwidth_reduction_ratio": float(bandwidth_reduction_ratio),
            "traditional_comparison": {
                "freeze_prevented": traditional_freeze,
                "crash_prevented": traditional_crash
            }
        }
