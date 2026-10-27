"""
L1_causal_field/core/embodied/embodied_sensorimotor_prediction_loop.py
========================================================================
Embodied Sensorimotor Prediction & Recalibration Loop (체화된 감각-운동 예측 및 재인식 루프)

Philosophy:
"상상과 예측이란 내면의 인과적 장력을 외부 세계로 던지는 '투영(Projection)과 투사(Transjection)'이다.
지능은 투영된 인과가 신체적 개입(Sensorimotor Intervention)을 통해 외부 세계의 실재 인과와 충돌하여
발생하는 낙차/마찰열(Cognitive Heat) 및 장력 파열(Rupture)을 온몸으로 감지하고,
오차와 분기적 요소(Branching Factor) 속에서 관계성 지형을 스스로 다듬고 재구축(Recalibration)할 때 비로소 탄생한다."

4-Stage Biological Sensorimotor Loop:
1. 투영 및 투사 (Projection & Transjection):
   - 내면의 인과 장력 필드(Φ_internal)를 바탕으로 외부 자극/환경에 대한 기대 파동 및 반응(P_exp) 투사.
2. 신체적 개입 (Sensorimotor Intervention):
   - 가상 신체 기질(Embodied Substrate / Organ Continuum)을 통해 행동/운동 벡터를 외부 표면에 밀어 넣음.
3. 충돌과 낙차 (Collision & Cognitive Heat):
   - 외부 실재 피드백(W_real)과 내면의 예측(P_exp) 사이의 위상 괴리(Δ = W_real - P_exp) 측정을 통해
     인지적 열(Q_heat = ||Δ|| * Tension) 및 장력 왜곡/파열(Rupture) 산출.
4. 분기적 재인식 (Branching & Recalibration):
   - Δ ≈ 0 (예측 일치): 기존 위상적 인과 경로를 바위처럼 고착 및 안정화 (Consolidation).
   - Δ > threshold (예측 불일치): 장력 왜곡 지점에서 새로운 분기 노드(Branching Factor) 발아,
     사물/타인과의 관계성 지형(Relational Topology) 재구축 및 인지 해상도 확장.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

from L1_causal_field.core.sensory.embodied_organ_continuum import (
    EmbodiedMultidimensionalOrganContinuum,
    MultidimensionalContinuumOutput,
)
from L1_causal_field.core.sensory.embodied_causal_substrate import (
    EmbodiedCausalSubstrate,
    EngramSymbol,
)


@dataclass
class ProjectionWave:
    """내면의 인과 장력이 외부에 던져진 투영/투사 파동"""
    projection_vector: np.ndarray        # 투영된 기대 파동 벡터
    internal_tension: float              # 내면의 장력 (Tension)
    intended_action_vector: np.ndarray   # 신체적 개입을 위해 준비된 행동 벡터
    causal_hypothesis: str              # "이렇게 행동하면 세상이 이렇게 반응할 것"이라는 가설


@dataclass
class SensorimotorCollisionResult:
    """투영된 예측과 신체적 개입이 세상의 실재 인과와 충돌한 결과"""
    expected_wave: ProjectionWave
    real_feedback_vector: np.ndarray     # 세상이 돌려준 진짜 반응/외력
    discrepancy_vector: np.ndarray       # 괴리 벡터 Δ = W_real - P_exp
    discrepancy_magnitude: float         # 괴리의 크기 ||Δ||
    cognitive_heat: float                # 인지적 열 Q_heat = ||Δ|| * Tension
    is_rupture: bool                     # 장력 파열 여부 (Cognitive Heat > Threshold)
    organ_discernment: MultidimensionalContinuumOutput # L0 다차원 감각 기관의 자율 분별


@dataclass
class RelationalRecalibrationOutput:
    """충돌 오차와 분기적 요소로부터 관계성 지형을 재구축한 결과"""
    collision: SensorimotorCollisionResult
    recalibrated_topology: np.ndarray    # 재구축된 위상적 인과 지도
    branching_factor_sprouted: bool      # 새로운 분기적 요소(Branching Node) 발아 여부
    sprouted_branch_id: Optional[str]    # 발아된 분기 노드 ID
    stabilization_score: float           # 기존 경로의 안정화/고착도 점수
    recalibration_narrative: str         # 오차를 수습하며 관계성을 재인식한 인과적 서사


class EmbodiedSensorimotorPredictionLoop(nn.Module):
    """
    체화된 감각-운동 예측 및 재인식 마스터 루프 엔진.
    """

    def __init__(
        self,
        vector_dim: int = 8,
        heat_threshold_rupture: float = 2.5,
        branching_threshold: float = 1.2,
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.heat_threshold_rupture = heat_threshold_rupture
        self.branching_threshold = branching_threshold

        # 신체 기질 및 L0 감각 기관 연속체 연동
        self.organ_continuum = EmbodiedMultidimensionalOrganContinuum(spatial_resolution=32)
        self.causal_substrate = EmbodiedCausalSubstrate(vector_dim=vector_dim)

        # 내면의 위상 인과 지형 (Relational Topology Matrix)
        self.relational_topology = np.eye(vector_dim, dtype=np.float64)
        self.internal_tension_state = 1.0
        self.branch_counter = 0

    def project_causal_expectation(
        self,
        internal_intent: np.ndarray,
        sensory_context: Optional[Any] = None,
    ) -> ProjectionWave:
        """
        1. 투영 및 투사 (Projection & Transjection):
        내면의 인과 장력 필드를 바탕으로 세상에 대한 가상의 기대 반응(P_exp)과
        개입 의도 벡터를 외부에 던짐.
        """
        arr_intent = np.asarray(internal_intent, dtype=np.float64).flatten()
        if len(arr_intent) < self.vector_dim:
            arr_intent = np.pad(arr_intent, (0, self.vector_dim - len(arr_intent)))
        else:
            arr_intent = arr_intent[:self.vector_dim]

        # 내면의 위상 지형(relational_topology)과의 곱을 통한 기대 투영 파동 생성
        projected_vector = np.dot(self.relational_topology, arr_intent)
        norm_p = np.linalg.norm(projected_vector)
        if norm_p > 1e-8:
            projected_vector = projected_vector / norm_p

        # 신체적 개입을 위한 운동/행동 벡터 (Action Vector)
        action_vector = np.tanh(projected_vector * self.internal_tension_state)

        hypothesis = (
            f"내면 장력({self.internal_tension_state:.2f})에 기반하여 "
            f"투영 파동(norm={norm_p:.2f})을 세상에 투사하고 행동 벡터를 준비합니다."
        )

        return ProjectionWave(
            projection_vector=projected_vector,
            internal_tension=self.internal_tension_state,
            intended_action_vector=action_vector,
            causal_hypothesis=hypothesis,
        )

    def execute_sensorimotor_intervention(
        self,
        projection: ProjectionWave,
        external_input_data: Any,
        modality_hint: str = "auto",
    ) -> SensorimotorCollisionResult:
        """
        2. 신체적 개입 & 3. 충돌과 낙차 (Collision & Cognitive Heat):
        투영된 행동을 외부 상에 밀어 넣어 L0 가상 감각 기관과 신체 기질에서
        실제 피드백(W_real)을 얻고, 예측과의 괴리(Δ) 및 인지적 열(Q_heat)을 측정.
        """
        # A. L0 다차원 감각 기관 연속체로 외부 자극 수용
        vis_in, ac_in, so_in = None, None, None

        # 문자열(수식/텍스트)인 경우 아스키 파동 변환하여 acoustic/somatosensory 입력 형성
        if isinstance(external_input_data, str):
            bytes_val = np.frombuffer(external_input_data.encode("utf-8"), dtype=np.uint8).astype(np.float64)
            ac_in = bytes_val
            so_in = bytes_val[:2] if len(bytes_val) >= 2 else np.pad(bytes_val, (0, 2 - len(bytes_val)))
        else:
            if modality_hint in ("image", "video") or isinstance(external_input_data, (np.ndarray, list)):
                vis_in = external_input_data
            if modality_hint in ("speech", "sound"):
                ac_in = external_input_data
            if modality_hint in ("tactile", "data"):
                so_in = external_input_data

            if vis_in is None and ac_in is None and so_in is None:
                vis_in = external_input_data

        organ_out = self.organ_continuum.process_multimodal_contact(
            visual_input=vis_in,
            acoustic_input=ac_in,
            somatosensory_input=so_in,
            resistance_dial=0.7,
            kenosis_level=0.5,
        )

        # B. 외부 실재 피드백 벡터(W_real) 산출
        # organ_out의 unified_phase_vector 및 organ 특성으로부터 물리적 피드백 구성
        base_feedback = np.zeros(self.vector_dim, dtype=np.float64)
        phases = organ_out.unified_phase_vector
        for i in range(min(len(phases), self.vector_dim)):
            base_feedback[i] = math.sin(phases[i])

        # 외부 입력 데이터 자체의 스펙트럼 에너지 반영
        if isinstance(external_input_data, np.ndarray):
            flat = external_input_data.flatten()
            for i in range(min(len(flat), self.vector_dim)):
                base_feedback[i] += 0.5 * flat[i]
        elif isinstance(external_input_data, str):
            # 텍스트/수식의 아스키/문자 파동 변환
            bytes_val = external_input_data.encode("utf-8")
            for i, b in enumerate(bytes_val[:self.vector_dim]):
                base_feedback[i] += math.cos(b / 255.0 * 2 * math.pi)

        norm_w = np.linalg.norm(base_feedback)
        if norm_w > 1e-8:
            real_feedback_vector = base_feedback / norm_w
        else:
            real_feedback_vector = base_feedback

        # C. 예측(P_exp)과 실재(W_real)의 괴리(Δ) 및 인지적 열(Q_heat) 계산
        discrepancy_vector = real_feedback_vector - projection.projection_vector
        discrepancy_mag = float(np.linalg.norm(discrepancy_vector))

        # 인지적 열: Q_heat = ||Δ|| * Tension
        cognitive_heat = discrepancy_mag * projection.internal_tension
        is_rupture = cognitive_heat > self.heat_threshold_rupture

        return SensorimotorCollisionResult(
            expected_wave=projection,
            real_feedback_vector=real_feedback_vector,
            discrepancy_vector=discrepancy_vector,
            discrepancy_magnitude=discrepancy_mag,
            cognitive_heat=cognitive_heat,
            is_rupture=is_rupture,
            organ_discernment=organ_out,
        )

    def recalibrate_relational_topology(
        self,
        collision: SensorimotorCollisionResult,
    ) -> RelationalRecalibrationOutput:
        """
        4. 분기적 재인식 (Branching & Recalibration):
        충돌 오차와 낙차로부터 기존 관계성 지형을 깎고 다듬으며 자율 재구축.
        - Δ ≈ 0: 기존 위상 경로 단단하게 고착/안정화.
        - Δ > threshold: 장력 파열 지점에서 새로운 분기 노드 발아 및 관계성 지형 재구축.
        """
        disc_mag = collision.discrepancy_magnitude
        heat = collision.cognitive_heat
        delta_vec = collision.discrepancy_vector

        sprouted_branch = False
        branch_id = None
        stabilization_score = 0.0

        if disc_mag < 0.4:
            # A. 예측 일치: 기존 경로 바위처럼 고착 및 안정화
            stabilization_score = float(1.0 - disc_mag)
            # 관계성 지형 강화 (외적 차원 공진)
            outer_prod = np.outer(collision.expected_wave.projection_vector, collision.real_feedback_vector)
            self.relational_topology += 0.05 * outer_prod
            # 장력 완화
            self.internal_tension_state = max(0.2, self.internal_tension_state * 0.9)

            narrative = (
                f"내면의 투영 예측과 세상의 실재 피드백이 높은 정확도(괴리 {disc_mag:.3f})로 일치했습니다. "
                f"기존의 위상적 경로가 바위처럼 단단하게 고착/안정화(점수 {stabilization_score:.2f})되었습니다."
            )

        else:
            # B. 예측 불일치 및 충돌: 새로운 분기 노드 발아 및 지형 재구축
            self.internal_tension_state = min(5.0, self.internal_tension_state + heat * 0.2)

            if heat > self.branching_threshold or collision.is_rupture:
                sprouted_branch = True
                self.branch_counter += 1
                branch_id = f"branch_node_{self.branch_counter}"

                # 오차(delta_vec) 방향으로 위상적 관계성 지형 재구축 (Plasticity Rewiring)
                correction_matrix = np.outer(delta_vec, collision.expected_wave.intended_action_vector)
                self.relational_topology += 0.15 * correction_matrix

                narrative = (
                    f"예측과 현실 사이에 거대한 충돌과 낙차(괴리 {disc_mag:.3f}, 인지적 열 {heat:.3f})가 발생했습니다! "
                    f"장력이 왜곡되며 새로운 분기 노드({branch_id})가 발아했고, "
                    f"오차를 수습하기 위해 내면의 관계성 지형(Relational Topology)이 한 차원 깊게 재구축되었습니다."
                )
            else:
                # 미세 오차 조율
                self.relational_topology += 0.05 * np.outer(delta_vec, delta_vec)
                narrative = (
                    f"중간 수준의 미세 오차(괴리 {disc_mag:.3f}, 인지적 열 {heat:.3f})가 감지되어 "
                    f"위상 지형의 장력을 유연하게 미세 조율했습니다."
                )

        # 위상 지형 정규화 (폭주 방지)
        norm_topo = np.linalg.norm(self.relational_topology)
        if norm_topo > 10.0:
            self.relational_topology = self.relational_topology / norm_topo * 10.0

        return RelationalRecalibrationOutput(
            collision=collision,
            recalibrated_topology=self.relational_topology.copy(),
            branching_factor_sprouted=sprouted_branch,
            sprouted_branch_id=branch_id,
            stabilization_score=stabilization_score,
            recalibration_narrative=narrative,
        )

    def run_full_sensorimotor_cycle(
        self,
        internal_intent: np.ndarray,
        external_input_data: Any,
        modality_hint: str = "auto",
    ) -> RelationalRecalibrationOutput:
        """
        4단계 전체 감각-운동 루프 (Projection -> Intervention -> Collision -> Recalibration) 단일 가동
        """
        # 1. 투영 및 투사
        projection = self.project_causal_expectation(
            internal_intent=internal_intent,
            sensory_context=external_input_data,
        )

        # 2 & 3. 신체적 개입 및 충돌 마찰열 측정
        collision = self.execute_sensorimotor_intervention(
            projection=projection,
            external_input_data=external_input_data,
            modality_hint=modality_hint,
        )

        # 4. 관계성 지형 재구축 및 분기
        recalibration = self.recalibrate_relational_topology(collision=collision)

        return recalibration
