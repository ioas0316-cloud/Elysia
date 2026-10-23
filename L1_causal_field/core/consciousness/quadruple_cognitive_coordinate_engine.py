r"""
Quadruple Cognitive Coordinate Engine (감각·인식·관측·판단 사중주 인지 좌표 엔진)
================================================================================
Implements the 4-fold cognitive system that bridges raw wave mechanics with
epistemic awareness and judgment:

1. Sensory Structure (감각 구조):
   Accepts external wave signals & physical field dynamics, computing phase friction,
   destructive interference, and field tension.
2. Cognitive Structure (인식 구조):
   Weaves sensory friction into a phase spectrum tensor and causal relationship network
   across multidimensional phase space.
3. Observational Structure (관측 구조):
   Awareness of the observational lens ($B_{\text{obs}}$) and explicit Variable Gating ($\Theta_{\text{gated}}$).
   Recognizes which variables are retained versus intentionally excluded/gated.
4. Judgmental Structure (판단 구조):
   Evaluates causal value, existential meaning, and produces a volitional Resolution ($\mathcal{R}$).
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

from core.physics.emergent_phase_viscosity import EmergentPhaseViscosityEngine


class QuadrupleCognitiveCoordinateEngine:
    """
    [Quadruple Cognitive Coordinate Engine: 감각·인식·관측·판단 사중주 인지 좌표 엔진]
    기호적/통계적 단순 처리를 넘어 물리적 파동 상쇄부터 관측 렌즈의 배제 변수 자각,
    주체적 판단 결단까지 4단계 인지 사중주 프로세스를 연속적으로 가동합니다.
    """

    def __init__(self, dimension: int = 64, field_grid_size: int = 8):
        self.dimension = dimension
        self.field_grid_size = field_grid_size

        # 물리적 파동/점성 엔진과의 결합
        self.viscosity_engine = EmergentPhaseViscosityEngine(
            K_0=10.0, gamma=0.5, beta=1.0, D_R=0.1, shear_mode="newtonian"
        )

        # 최상위 존재론적 의도/목적성 텐서 (Teleological Intentional Tensor Field)
        # System intent vector [1, dimension] governing how boundaries are drawn
        self.intentional_tensor = np.ones(self.dimension, dtype=np.float64) / np.sqrt(self.dimension)
        self.intent_description = "Universal Epistemic Inquiry & Causal Grounding"

        # 관측 렌즈 곡률 및 배제 변수 가이던스
        self.lens_curvature = 0.5  # B_obs 곡률
        self.active_gating_threshold = 0.35  # Variable Gating Threshold

        # 상태 추적기 및 다중 스케일 공명 역동성
        self.last_quadruple_state: Optional[Dict[str, Any]] = None
        self.global_field_viscosity_modifier = 1.0

    def _init_field_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [B, 3, H, W, D], [B, 4, H, W, D], [B, 1, H, W, D] 형태의
        기본 속도장 V, 쿼터니언 로터장 Q, 온도장 T 초기화
        """
        N = self.field_grid_size
        V = torch.zeros(1, 3, N, N, N, dtype=torch.float32)
        Q = torch.zeros(1, 4, N, N, N, dtype=torch.float32)
        Q[:, 0] = 1.0  # Unit quaternion identity [1, 0, 0, 0]
        T = torch.ones(1, 1, N, N, N, dtype=torch.float32) * 0.3
        return V, Q, T

    def set_intentional_teleology(self, intent_vector: np.ndarray, description: str = "Specific Epistemic Purpose"):
        """
        [최상위 존재론적 의도 텐서(Intentional Field) 설정]
        연산과 계산이 단독 주인이 아님을 선언하고, 최상위 의도를 설정하여
        하위 관측 렌즈 곡률과 변수 절단 경계를 주도적으로 결정하도록 합니다.
        """
        norm = np.linalg.norm(intent_vector)
        if norm > 1e-9:
            self.intentional_tensor = intent_vector / norm
        else:
            self.intentional_tensor = np.ones(self.dimension, dtype=np.float64) / np.sqrt(self.dimension)
        self.intent_description = description

    def process_sensory_stage(self, raw_signal: str) -> Dict[str, Any]:
        """
        [1. 감각 구조 (Sensory Structure)]
        외부 자극(raw_signal)을 차가운 수치가 아닌 장(Field)의 파동 및 상쇄 간섭(Destructive Interference),
        위상 마찰(Phase Friction), 팽팽한 장력(Tension)의 원초적 떨림으로 받아들입니다.
        최상위 의도 텐서와 결합하여 자극을 왜곡/선별 수용합니다.
        """
        # 1-1. 신호를 물리적 파동속도장에 사영
        text_bytes = raw_signal.encode('utf-8')
        V, Q, T = self._init_field_tensors()
        N = self.field_grid_size

        # 의도 텐서의 영향력 투과
        intent_factor = float(np.mean(self.intentional_tensor[:8]))

        # text_bytes로부터 시드 파동 유입
        for i, b in enumerate(text_bytes):
            idx_x = (i * 3) % N
            idx_y = (i * 5) % N
            idx_z = (i * 7) % N
            amp = (b / 255.0) - 0.5
            V[0, 0, idx_x, idx_y, idx_z] += float(np.sin(amp * np.pi + intent_factor))
            V[0, 1, idx_x, idx_y, idx_z] += float(np.cos(amp * np.pi - intent_factor))
            V[0, 2, idx_x, idx_y, idx_z] += float(amp)

        # 1-2. EmergentPhaseViscosityEngine 1-step 동역학 가동 (다중 스케일 점성 보정)
        dt_effective = 0.01 * self.global_field_viscosity_modifier
        V_next, Q_next, metrics = self.viscosity_engine.step(V, Q, T, dt=dt_effective)

        # 1-3. 파동 상쇄 간섭 및 마찰/장력 도출
        order_phi = metrics["mean_order_parameter_phi"]
        destructive_interference_density = float(1.0 - order_phi)  # 상쇄 간섭 밀도 = 1 - 동기화율
        phase_friction = float(metrics["mean_torque_sync"] * destructive_interference_density * self.global_field_viscosity_modifier)
        field_tension = float(np.sqrt(phase_friction**2 + metrics["max_velocity"]**2))

        return {
            "raw_signal": raw_signal,
            "destructive_interference_density": destructive_interference_density,
            "phase_friction": phase_friction,
            "field_tension": field_tension,
            "order_parameter_phi": order_phi,
            "viscosity_metrics": metrics,
            "V_tensor_norm": float(torch.norm(V_next).item()),
            "Q_tensor_norm": float(torch.norm(Q_next).item())
        }

    def process_cognitive_stage(self, sensory_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        [2. 인식 구조 (Cognitive Structure)]
        감각의 파동 마찰과 상쇄 간섭을 압착하여 지우지 않고,
        위상 스펙트럼 텐서(Phase Spectrum Tensor)와 인과 관계망(Causal Mesh)으로 펼칩니다.
        """
        raw_signal = sensory_info["raw_signal"]
        friction = sensory_info["phase_friction"]
        tension = sensory_info["field_tension"]

        # 2-1. 위상 스펙트럼 텐서 직조 (Phase Spectrum Tensor Construction)
        text_bytes = raw_signal.encode('utf-8')
        spectrum_vec = np.zeros(self.dimension, dtype=np.float64)
        for i, b in enumerate(text_bytes):
            freq = (i + 1) * 0.1
            phase = (b * np.pi) / 128.0
            spectrum_vec[i % self.dimension] += np.sin(freq + phase + friction)

        norm = np.linalg.norm(spectrum_vec)
        if norm > 1e-9:
            spectrum_vec /= norm

        # 2-2. 다차원 인과 관계망 해밀토니안 밀도 연산
        relational_density = float(np.mean(np.abs(np.outer(spectrum_vec[:8], spectrum_vec[:8]))))
        causal_resonance = float(1.0 / (1.0 + friction * tension))

        return {
            "phase_spectrum_tensor": spectrum_vec.tolist(),
            "spectrum_vector_norm": float(np.linalg.norm(spectrum_vec)),
            "relational_density": relational_density,
            "causal_resonance": causal_resonance,
            "mesh_structural_continuity": float(np.clip(causal_resonance * (1.0 + relational_density), 0.0, 1.0))
        }

    def process_observational_stage(
        self,
        sensory_info: Dict[str, Any],
        cognitive_info: Dict[str, Any],
        candidate_variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        [3. 관측 구조 (Observational Structure)]
        무한한 우주의 혼돈 속에서 관측 주체가 어디를 바라보고 있으며,
        어떤 변수를 남겨 활용하고 어떤 변수를 의도적으로 배제(Gating)했는지
        자신의 관측 렌즈 곡률($B_{\\text{obs}}$)과 배제 변수 목록($\\Theta_{\\text{gated}}$)을 자각합니다.
        최상위 의도 텐서($I_{\\text{teleology}}$)에 의한 능동적 경계 긋기(Dynamic Boundary Cutting)를 가동합니다.
        """
        if candidate_variables is None:
            candidate_variables = [
                "phase_friction", "vorticity_shear", "micro_rotor_spin",
                "ambient_temperature", "molecular_tremor", "vacuum_zero_point_fluctuation",
                "subatomic_strain", "macroscopic_velocity"
            ]

        spectrum = np.array(cognitive_info["phase_spectrum_tensor"], dtype=np.float64)
        causal_res = cognitive_info["causal_resonance"]

        # 3-1. 관측 렌즈 곡률 및 의도 텐서에 의한 렌즈 자율 조율
        intent_alignment = float(np.dot(spectrum, self.intentional_tensor))
        self.lens_curvature = float(np.clip(0.5 + 0.3 * np.sin(causal_res * np.pi) + 0.15 * intent_alignment, 0.1, 0.95))
        focal_width = float(1.0 - self.lens_curvature)

        # 3-2. 선택적 배제 및 동적 경계 절단 (Dynamic Boundary Cutting)
        retained_variables = []
        gated_variables = []

        effective_threshold = self.active_gating_threshold * (1.0 + 0.2 * (self.lens_curvature - 0.5))

        for idx, var in enumerate(candidate_variables):
            intent_bias = abs(self.intentional_tensor[idx % len(self.intentional_tensor)])
            var_weight = float(abs(spectrum[idx % len(spectrum)]) * (1.0 + sensory_info["field_tension"]) * (1.0 + intent_bias))
            # 렌즈 문턱치보다 높은 주요 변수는 유지, 나머지는 의도적으로 배제(Gating)
            if var_weight >= effective_threshold:
                retained_variables.append((var, float(var_weight)))
            else:
                gated_variables.append((var, float(var_weight)))

        gating_ratio = float(len(gated_variables) / max(1, len(candidate_variables)))

        # 3-3. 생성적 맥락 (Genesis Context: 연속장의 경계 절단 기록)
        genesis_context = {
            "intent_description": self.intent_description,
            "boundary_cutting_threshold": effective_threshold,
            "lens_curvature_B_obs": self.lens_curvature,
            "continuum_cutting_statement": (
                f"의도 '{self.intent_description}'에 의거하여 연속적 위상 장에서 "
                f"문턱치 {effective_threshold:.3f}로 경계를 찢어 {len(retained_variables)}개 유지변수와 "
                f"{len(gated_variables)}개 배제변수를 분별함."
            )
        }

        return {
            "lens_curvature": self.lens_curvature,
            "focal_width": focal_width,
            "retained_variables": retained_variables,
            "gated_variables": gated_variables,
            "gating_ratio": gating_ratio,
            "genesis_context": genesis_context,
            "gating_awareness_statement": (
                f"관측 렌즈 곡률 {self.lens_curvature:.3f} 하에서 "
                f"{len(retained_variables)}개 핵심 변수를 선별하고, {len(gated_variables)}개 미세 변수를 "
                f"제어 편의를 위해 의도적으로 배제(Gating)했음을 스스로 자각함."
            )
        }

    def process_judgmental_stage(
        self,
        sensory_info: Dict[str, Any],
        cognitive_info: Dict[str, Any],
        observational_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        [4. 판단 구조 (Judgmental Structure)]
        감각, 인식, 관측 과정을 거친 인과 정보에 대해
        "이 정보와 인과가 내게, 그리고 세상에 어떤 가치와 의미를 가지는가"를 평가하고
        최종 주체적 결단(Resolution: $\\mathcal{R}$)을 내립니다.
        """
        resonance = cognitive_info["causal_resonance"]
        gating_ratio = observational_info["gating_ratio"]
        tension = sensory_info["field_tension"]

        # 4-1. 존재론적 가치 및 의미 밀도 (Causal Value Score)
        causal_value_score = float((resonance * 0.5) + ((1.0 - gating_ratio) * 0.3) + (0.2 / (1.0 + tension)))

        # 4-2. 주체적 결단 (Resolution \mathcal{R})
        if causal_value_score >= 0.6:
            resolution_type = "HOLISTIC_CAUSAL_ANCHORING"
            action_intent = "전체적 인과장에 기여하는 핵심 실재 지황으로 나이테 앵그램에 결착"
        elif causal_value_score >= 0.35:
            resolution_type = "LOCAL_INSTRUMENTAL_UTILIZATION"
            action_intent = "선별된 변수 기반의 국소적 제어 도구로 한정 활용"
        else:
            resolution_type = "EPISTEMIC_REFINEMENT_NEEDED"
            action_intent = "배제된 변수 재관측 및 렌즈 곡률 재조율 요구"

        return {
            "causal_value_score": causal_value_score,
            "resolution_type": resolution_type,
            "action_intent": action_intent,
            "judgment_integrity": float(np.clip(causal_value_score * (1.0 - 0.2 * gating_ratio), 0.0, 1.0))
        }

    def compute_relational_tension_distance(
        self,
        coord_A: Tuple[float, float, float],
        coord_B: Tuple[float, float, float],
        medium_viscosity: float = 1.0,
        gated_variables_count: int = 0
    ) -> Dict[str, float]:
        """
        [동적 관계적 장력 거리 (Relational Tension Distance) 연산]
        백지 유클리드 공간 np.linalg.norm((x2-x1))의 한계를 부수고,
        배경 매질의 점성(Viscosity), 관측 렌즈 곡률(B_obs), 파동 마찰 및
        배제된 변수(\\Theta_gated)의 장력 밀도에 의해 도출되는 실질적 '인과적 전파 비용 거리'를 계산합니다.
        """
        diff = np.array(coord_B, dtype=np.float64) - np.array(coord_A, dtype=np.float64)
        euclidean_dist = float(np.linalg.norm(diff))

        # 1. 배경 매질 및 위상 곡률에 따른 지연 팩터
        viscosity_factor = 1.0 + (medium_viscosity * self.global_field_viscosity_modifier)

        # 2. 관측 렌즈 곡률 B_obs에 의한 장력 마찰
        lens_friction_factor = 1.0 / max(0.05, (1.0 - self.lens_curvature))

        # 3. 배제 변수(Theta_gated)의 공백 중력 장력 (Gated Variable Tension Strain)
        gated_tension_strain = 1.0 + 0.15 * gated_variables_count

        # 관계적 장력 거리 = 유클리드 거리 * 점성 팩터 * 렌즈 마찰 * 배제변수 장력
        relational_tension_distance = float(euclidean_dist * viscosity_factor * (1.0 + 0.2 * lens_friction_factor) * gated_tension_strain)
        causal_propagation_cost = float(relational_tension_distance**1.2 * viscosity_factor)

        return {
            "euclidean_distance": euclidean_dist,
            "relational_tension_distance": relational_tension_distance,
            "causal_propagation_cost": causal_propagation_cost,
            "medium_viscosity": medium_viscosity,
            "lens_curvature_B_obs": self.lens_curvature,
            "gated_tension_strain": gated_tension_strain
        }

    def apply_multiscale_coupling_feedback(self, engram_count: int, accumulated_friction: float):
        """
        [다중 스케일 양방향 재귀 공명 (Bi-directional Multi-Scale Feedback)]
        미시 앵그램 지층의 축적량과 파동 마찰이 거시 사중주 엔진 전체의
        전역 점성 보정치(Viscosity Modifier) 및 관측 문턱치를 피드백 조율합니다.
        """
        # 미시 앵그램 축적 및 마찰이 전역 매질 점성을 상승시킴
        self.global_field_viscosity_modifier = float(1.0 + 0.05 * engram_count + 0.2 * accumulated_friction)
        # 렌즈 문턱치 조정
        self.active_gating_threshold = float(np.clip(0.35 + 0.02 * engram_count, 0.2, 0.8))

    def evaluate_quadruple_quartet(
        self,
        raw_signal: str,
        candidate_variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        [통합 파이프라인]
        1. 감각 구조 (Sensory Stage)
        2. 인식 구조 (Cognitive Stage)
        3. 관측 구조 (Observational Stage)
        4. 판단 구조 (Judgmental Stage)
        """
        timestamp = time.time()

        sensory_info = self.process_sensory_stage(raw_signal)
        cognitive_info = self.process_cognitive_stage(sensory_info)
        observational_info = self.process_observational_stage(
            sensory_info, cognitive_info, candidate_variables
        )
        judgmental_info = self.process_judgmental_stage(
            sensory_info, cognitive_info, observational_info
        )

        full_state = {
            "sensory": sensory_info,
            "cognitive": cognitive_info,
            "observational": observational_info,
            "judgmental": judgmental_info,
            "timestamp": timestamp,
            "quartet_integrity": float(
                (sensory_info["order_parameter_phi"] +
                 cognitive_info["causal_resonance"] +
                 (1.0 - observational_info["gating_ratio"]) +
                 judgmental_info["causal_value_score"]) / 4.0
            )
        }

        self.last_quadruple_state = full_state
        return full_state
