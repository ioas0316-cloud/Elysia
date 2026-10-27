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
    """내면의 인과 장력이 외부에 던져진 투영/투사 파동 (Step 1)"""
    projection_vector: np.ndarray        # 투영된 기대 파동 벡터 (P_exp)
    internal_tension: float              # 내면의 장력 (Tension)
    intended_action_vector: np.ndarray   # 신체적 개입을 위해 준비된 행동 벡터
    thought_gradient: float              # 외력을 마주하기 전 사유의 기울기 (Gradient of Thought)
    causal_hypothesis: str              # "이렇게 행동하면 세상이 이렇게 반응할 것"이라는 가설


@dataclass
class SensorimotorInterventionData:
    """가상 신체 기질이 입력 데이터를 외력 벡터 및 물리적 압력으로 환산한 결과 (Step 2)"""
    external_force_vector: np.ndarray    # 외력 벡터 (External Force Vector)
    physical_pressure_value: float       # 물리적 압력 값 (Morphology/Pixel variation rate)
    energy_density_drop: float           # 에너지 밀도 낙차
    modality_hint: str


@dataclass
class SensorimotorCollisionResult:
    """투영된 예측과 신체적 개입이 세상의 실재 인과와 충돌한 결과 (Step 3)"""
    expected_wave: ProjectionWave
    intervention: SensorimotorInterventionData # 신체적 개입 데이터
    real_feedback_vector: np.ndarray     # 세상이 돌려준 진짜 반응/외력 (W_real)
    discrepancy_vector: np.ndarray       # 괴리 벡터 Δ = W_real - P_exp
    discrepancy_magnitude: float         # 괴리의 크기 ||Δ||
    cognitive_heat: float                # 비선형 인지적 열 Q_heat = ||Δ|| * Tension * (1 + ||Δ||^2)
    is_rupture: bool                     # 장력 파열 여부 (Cognitive Heat > Threshold)
    organ_discernment: MultidimensionalContinuumOutput # L0 다차원 감각 기관의 자율 분별


@dataclass
class RelationalRecalibrationOutput:
    """충돌 오차와 분기적 요소로부터 관계성 지형을 재구축한 결과 (Step 4)"""
    collision: SensorimotorCollisionResult
    recalibrated_topology: np.ndarray    # 재구축된 위상적 인과 지도
    branching_factor_sprouted: bool      # 새로운 분기적 요소(Branching Node) 발아 여부
    sprouted_branch_id: Optional[str]    # 발아된 분기 노드 ID
    stabilization_score: float           # 기존 경로의 안정화/고착도 점수
    causal_step_sequence: List[str]      # 4단계 인과서순 실행 타임라인 스냅샷
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
        Step 1: 투영 및 투사 (Projection Phase)
        - 입력 데이터가 들어오기 전, 시스템 내부의 현재 장력 필드(Φ_internal)가 예상하는 기대 반응 파동(P_exp) 형성.
        - 외력을 마주하기 전 품고 있던 '사유의 기울기(Gradient of Thought)'를 스냅샷으로 기록.
        """
        arr_intent = np.asarray(internal_intent, dtype=np.float64).flatten()
        if len(arr_intent) < self.vector_dim:
            arr_intent = np.pad(arr_intent, (0, self.vector_dim - len(arr_intent)))
        else:
            arr_intent = arr_intent[:self.vector_dim]

        # 내면의 위상 지형(relational_topology)과의 곱을 통한 기대 투영 파동(P_exp) 생성
        projected_vector = np.dot(self.relational_topology, arr_intent)
        norm_p = np.linalg.norm(projected_vector)
        if norm_p > 1e-8:
            projected_vector = projected_vector / norm_p

        # 사유의 기울기 스냅샷 (Gradient of Thought: ||d(Relational Topology * intent)/d_intent||)
        thought_gradient = float(np.linalg.norm(self.relational_topology) * self.internal_tension_state)

        # 신체적 개입을 위한 운동/행동 벡터 (Action Vector)
        action_vector = np.tanh(projected_vector * self.internal_tension_state)

        hypothesis = (
            f"Step 1 [투영]: 내면 장력({self.internal_tension_state:.2f})과 사유의 기울기({thought_gradient:.2f})에 기반하여 "
            f"투영 파동 P_exp(norm={norm_p:.2f})를 공간에 형성합니다."
        )

        return ProjectionWave(
            projection_vector=projected_vector,
            internal_tension=self.internal_tension_state,
            intended_action_vector=action_vector,
            thought_gradient=thought_gradient,
            causal_hypothesis=hypothesis,
        )

    def execute_sensorimotor_intervention(
        self,
        projection: ProjectionWave,
        external_input_data: Any,
        modality_hint: str = "auto",
    ) -> SensorimotorCollisionResult:
        """
        Step 2: 신체적 개입 및 마찰 유발 (Intervention & Friction Phase)
        - 가상 신체 기질(Sensorimotor Continuum)이 입력 데이터를 픽셀이나 텍스트가 아닌 '외력 벡터(External Force Vector)'로 전환.
        - 입력 데이터의 형태소/픽셀 변화율을 물리적 압력 값(Physical Pressure Value) 및 에너지 밀도 낙차로 환산.

        Step 3: 충돌과 인지적 열 측정 (Collision & Cognitive Heat Generation Phase)
        - 실제 피드백(W_real)과 예측(P_exp) 사이의 위상 괴리(Δ = W_real - P_exp) 계산.
        - 비선형 인지적 열: Q_heat = ||Δ|| * Tension * (1 + ||Δ||^2) 도초 및 장력-마찰 스펙트럼 축 위 파열(Rupture) 판정.
        """
        # Step 2: 외력 벡터(External Force Vector) 및 물리적 압력 환산
        ext_force = np.zeros(self.vector_dim, dtype=np.float64)
        physical_pressure = 0.0
        energy_density_drop = 0.0

        if isinstance(external_input_data, np.ndarray):
            flat = external_input_data.astype(np.float64).flatten()
            std_val = float(np.std(flat)) if len(flat) > 0 else 0.1
            mean_val = float(np.mean(np.abs(flat))) if len(flat) > 0 else 0.1
            physical_pressure = float(std_val * 2.5 + mean_val)
            energy_density_drop = float(np.max(flat) - np.min(flat)) if len(flat) > 0 else 1.0

            for i in range(min(len(flat), self.vector_dim)):
                ext_force[i] = flat[i]

        elif isinstance(external_input_data, str):
            bytes_val = np.frombuffer(external_input_data.encode("utf-8"), dtype=np.uint8).astype(np.float64)
            char_diffs = np.abs(np.diff(bytes_val)) if len(bytes_val) > 1 else np.array([1.0])
            physical_pressure = float(np.mean(char_diffs) / 10.0 + len(bytes_val) * 0.05)
            energy_density_drop = float(np.std(bytes_val) / 5.0) if len(bytes_val) > 1 else 0.5

            for i, b in enumerate(bytes_val[:self.vector_dim]):
                ext_force[i] = math.cos(b / 255.0 * 2 * math.pi)

        norm_ext = np.linalg.norm(ext_force)
        if norm_ext > 1e-8:
            ext_force = ext_force / norm_ext

        intervention_data = SensorimotorInterventionData(
            external_force_vector=ext_force,
            physical_pressure_value=physical_pressure,
            energy_density_drop=energy_density_drop,
            modality_hint=modality_hint,
        )

        # L0 다차원 감각 기관 연속체 수용
        vis_in, ac_in, so_in = None, None, None
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

        # 실재 피드백 벡터(W_real) 결합 (기존 L0 + 외력 벡터)
        base_feedback = np.zeros(self.vector_dim, dtype=np.float64)
        phases = organ_out.unified_phase_vector
        for i in range(min(len(phases), self.vector_dim)):
            base_feedback[i] = math.sin(phases[i])

        base_feedback += 0.6 * ext_force
        norm_w = np.linalg.norm(base_feedback)
        real_feedback_vector = base_feedback / norm_w if norm_w > 1e-8 else base_feedback

        # Step 3: 충돌 및 괴리(Δ = W_real - P_exp), 비선형 인지적 열(Q_heat) 산출
        discrepancy_vector = real_feedback_vector - projection.projection_vector
        discrepancy_mag = float(np.linalg.norm(discrepancy_vector))

        # 비선형 열 발생: Q_heat = ||Δ|| * Tension * (1 + ||Δ||^2)
        cognitive_heat = discrepancy_mag * projection.internal_tension * (1.0 + discrepancy_mag ** 2)
        is_rupture = cognitive_heat > self.heat_threshold_rupture

        return SensorimotorCollisionResult(
            expected_wave=projection,
            intervention=intervention_data,
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
        Step 4: 분기 및 재인식 (Branching & Recalibration Phase)
        - Δ ≈ 0 (예측 부합): 기존 인과 경로의 가중치를 바위처럼 고착화 (Stabilization / Consolidation).
        - Δ > 임계값 (예측 파열): 장력 왜곡 지점에서 새로운 분기 노드(Branching Factor) 발아, 내면 관계성 지형 재구축.
        """
        disc_mag = collision.discrepancy_magnitude
        heat = collision.cognitive_heat
        delta_vec = collision.discrepancy_vector

        sprouted_branch = False
        branch_id = None
        stabilization_score = 0.0

        step_sequence = [
            f"1. [투영] P_exp 생성 (사유 기울기={collision.expected_wave.thought_gradient:.2f})",
            f"2. [개입] 외력 벡터 전환 (물리적 압력={collision.intervention.physical_pressure_value:.2f})",
            f"3. [충돌] 괴리 ||Δ||={disc_mag:.3f}, 비선형 Q_heat={heat:.3f} (파열={collision.is_rupture})",
        ]

        if disc_mag < 0.4:
            # 예측 부합: 고착화
            stabilization_score = float(1.0 - disc_mag)
            outer_prod = np.outer(collision.expected_wave.projection_vector, collision.real_feedback_vector)
            self.relational_topology += 0.05 * outer_prod
            self.internal_tension_state = max(0.2, self.internal_tension_state * 0.9)

            step_sequence.append(f"4. [재인식] 예측 부합 -> 기존 위상 경로 바위처럼 고착화 (점수={stabilization_score:.2f})")
            narrative = (
                f"투영 예측과 세상의 피드백이 높은 정합성(괴리 {disc_mag:.3f})으로 일치했습니다. "
                f"기존의 인과 경로가 고착화(안정화 점수 {stabilization_score:.2f})되었습니다."
            )
        else:
            # 예측 불일치 및 충돌: 분기 및 지형 재구축
            self.internal_tension_state = min(5.0, self.internal_tension_state + heat * 0.2)

            if heat > self.branching_threshold or collision.is_rupture:
                sprouted_branch = True
                self.branch_counter += 1
                branch_id = f"branch_node_{self.branch_counter}"

                correction_matrix = np.outer(delta_vec, collision.expected_wave.intended_action_vector)
                self.relational_topology += 0.15 * correction_matrix

                step_sequence.append(f"4. [재인식] 장력 파열(Q_heat={heat:.3f}) -> 신규 분기 노출 ({branch_id}) 및 위상 지형 재구축")
                narrative = (
                    f"거대한 충돌과 낙차(괴리 {disc_mag:.3f}, 인지적 열 {heat:.3f})로 인해 장력 파열이 발생했습니다! "
                    f"새로운 분기 노드({branch_id})가 발아했고 내면의 관계성 지형(Relational Topology)이 한 차원 깊게 재구축되었습니다."
                )
            else:
                self.relational_topology += 0.05 * np.outer(delta_vec, delta_vec)
                step_sequence.append(f"4. [재인식] 미세 오차(Q_heat={heat:.3f}) -> 위상 지형 미세 조율")
                narrative = (
                    f"미세 오차(괴리 {disc_mag:.3f}, 인지적 열 {heat:.3f})가 감지되어 위상 지형의 장력을 미세 조율했습니다."
                )

        norm_topo = np.linalg.norm(self.relational_topology)
        if norm_topo > 10.0:
            self.relational_topology = self.relational_topology / norm_topo * 10.0

        return RelationalRecalibrationOutput(
            collision=collision,
            recalibrated_topology=self.relational_topology.copy(),
            branching_factor_sprouted=sprouted_branch,
            sprouted_branch_id=branch_id,
            stabilization_score=stabilization_score,
            causal_step_sequence=step_sequence,
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
        # Step 1: 투영 및 투사
        projection = self.project_causal_expectation(
            internal_intent=internal_intent,
            sensory_context=external_input_data,
        )

        # Step 2 & 3: 신체적 개입, 외력 환산 및 충돌/마찰열 측정
        collision = self.execute_sensorimotor_intervention(
            projection=projection,
            external_input_data=external_input_data,
            modality_hint=modality_hint,
        )

        # Step 4: 관계성 지형 재구축 및 분기
        recalibration = self.recalibrate_relational_topology(collision=collision)

        return recalibration
