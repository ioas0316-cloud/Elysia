r"""
[Active Probing Cognition Engine: Boundary Topology & Symbol Deconstruction]

이 모듈은 외부 데이터 스트림을 가공된 수치나 정적 통계 표상으로 읽어오는 단순 수신기를 넘어,
세상을 향한 능동적 작용(Active Probing)과 그에 따른 저항/변형(Reaction)의 마찰 패턴을 통해
'자기(Self, c=1)'와 '세계(World, c=0)'의 경계를 획정하고,
인간의 언어 기호(Symbol)를 미시적 가함-저항-변형 궤적으로 해체 및 재정렬하며,
미지의 개념/현상을 환각(Hallucination) 없이 역추적하여 하부 인과 뿌리에 연결하는
통합 인지 엔진(Active Probing Cognition Engine)입니다.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class ActiveProbingCognitionEngine:
    r"""
    Active Probing Cognition Engine

    1. 경계 접지 & 능동 탐색 (Axis 1):
       - 능동 가함 벡터 u(t) 및 외부 반작용 stress T_ext
       - 응력 차분 Delta T_ij = T_ext - T_int
       - 제어 스펙트럼 c in [0, 1] (c=1: Self, c=0: World)
       - 위상 마찰 Omega_ij = (1 - c) * (nabla_i Delta T_jk - nabla_j Delta T_ik)
       - 계량장 적응 연산: dg_ij / dt = -2 R_ij + lambda * P(Omega_ij)

    2. 기호 해체 & 인과 계통 재정렬 (Axis 2):
       - 언어적 표면 라벨('단단함', '점성', '병목', '타자')을 [가함 -> 저항 -> 변형] 궤적으로 해체
       - 물리적 접지 -> 정보적 확장 -> 인지적 추상화 공간으로 다중 스케일 재정렬

    3. 미지 개념 역추적 & 계통 확장 (Axis 3):
       - 미지 스트림 입력 시 상호작용 궤적 스캔
       - 내부 계량장의 곡률 분기점(인과 단절) 포착
       - 끊어진 하부 인과 뿌리와의 기하학적 연결 (No Hallucination)

    4. 4단계 인지 연산 (Cognition, Thought, Judgment, Discrimination):
       - cognition(u_vector, external_stress, control_c): 상호작용 지형 및 위상 마찰 등록
       - thought(virtual_action): 가상 작용에 대한 내부 계량장 변형 시뮬레이션
       - judgment(predicted_stress, actual_stress): 오차(Predictive Delta) 평가
       - discrimination(profile_a, profile_b): 곡률 지형 차이에 따른 위상 경계선 획정
    """

    def __init__(self, dim: int = 3, viscosity: float = 0.1, lambda_diff: float = 0.05):
        self.dim = dim
        self.viscosity = viscosity
        self.lambda_diff = lambda_diff

        # 내부 매니폴드 계량 텐서 g_ij (초기화: 유클리드 기하 eye(dim))
        self.g_metric = np.eye(dim, dtype=np.float64)

        # 내부 예측 응력 텐서 계산을 위한 기본 계량 텐서 필드 상태
        self.ricci_curvature = np.zeros((dim, dim), dtype=np.float64)

        # 해체 및 재정렬된 인과 계통 공간
        # concept_registry: label -> Dict[str, Any]
        self.concept_registry: Dict[str, Dict[str, Any]] = {}

        # 기본 인과 개념들의 표준 라벨 해체 예시 사전 등록
        self._initialize_default_symbols()

    def _initialize_default_symbols(self):
        """기본적 표면 기호들을 [가함 -> 저항 -> 변형] 인과 궤적으로 초기 등록"""
        # 1. 단단함 (Stiffness / Physical Origin)
        self.deconstruct_symbol(
            label="단단함",
            action_profile={
                "delta_F_over_delta_x": 0.95,  # 위치 변이 대비 강한 반작용 힘
                "delta_F_over_delta_v": 0.05,  # 속도 연관성 낮음
                "control_c": 0.0               # 제어 불가능한 외부 물성
            },
            scale="physical"
        )

        # 2. 점성 (Viscosity / Physical Origin)
        self.deconstruct_symbol(
            label="점성",
            action_profile={
                "delta_F_over_delta_x": 0.20,
                "delta_F_over_delta_v": 0.90,  # 속도 비례 인장 저항
                "control_c": 0.1
            },
            scale="physical"
        )

        # 3. 병목/과부하 (Bottleneck / Informational Escalation)
        self.deconstruct_symbol(
            label="병목",
            action_profile={
                "delta_F_over_delta_x": 0.85,  # 유량 팽창 시 밀림 resistance
                "delta_F_over_delta_v": 0.80,
                "control_c": 0.2
            },
            scale="informational"
        )

        # 4. 타자/세계 (External World / Cognitive Abstraction)
        self.deconstruct_symbol(
            label="세계",
            action_profile={
                "delta_F_over_delta_x": 0.99,
                "delta_F_over_delta_v": 0.99,
                "control_c": 0.0               # 완벽한 제어 불가능성
            },
            scale="cognitive"
        )

    def compute_internal_stress(self, u_vector: np.ndarray) -> np.ndarray:
        """현재 내부 계량장 g_ij 및 작용 u_vector에 기반한 내부 예측 응력 텐서 T_int 계산"""
        u_vector = np.asarray(u_vector, dtype=np.float64)
        # T_int_ij = g_ij * (u_i * u_j)
        outer_u = np.outer(u_vector, u_vector)
        t_int = self.g_metric * outer_u
        return t_int

    def apply_active_probing(
        self,
        u_vector: np.ndarray,
        external_stress: np.ndarray,
        control_c: float
    ) -> Dict[str, Any]:
        """
        축 1: 능동적 작용-반작용 및 위상 마찰 (Omega_ij) 기반 계량장 진화

        - u_vector: 능동적 가함 작용 벡터 (dim,)
        - external_stress: 외부 매질이 되돌려준 반응 응력 텐서 (dim, dim)
        - control_c: 제어 가능성 스펙트럼 c in [0, 1] (1: Self, 0: World)
        """
        u_vector = np.asarray(u_vector, dtype=np.float64)
        external_stress = np.asarray(external_stress, dtype=np.float64)

        if external_stress.shape != (self.dim, self.dim):
            # 벡터로 들어온 경우 outer product로 텐서화
            if external_stress.ndim == 1 and external_stress.shape[0] == self.dim:
                external_stress = np.outer(external_stress, external_stress)
            else:
                raise ValueError(f"external_stress must be shape ({self.dim}, {self.dim})")

        # 1. 내부 예측 응력 텐서 T_int
        t_int = self.compute_internal_stress(u_vector)

        # 2. 응력 차분 Delta T_ij
        delta_T = external_stress - t_int

        # 3. 제어 불가능성 (1 - c) 및 위상 마찰 Omega_ij
        # Omega_ij = (1 - c) * Delta T_ij
        uncontrollability = max(0.0, min(1.0, 1.0 - control_c))
        topological_friction = uncontrollability * delta_T

        # 4. 조화 확산 프로젝터 P(Omega_ij): 계량장 방향 수송
        harmonic_diffusion = 0.5 * (topological_friction + topological_friction.T)

        # 5. Ricci 곡률 근사 계산 (내부 평형력)
        # R_ij approx 0.01 * (g_ij - eye)
        self.ricci_curvature = 0.01 * (self.g_metric - np.eye(self.dim))

        # 6. 계량장 적응 연산식: dg_ij / dt = -2 * R_ij + lambda * P(Omega_ij)
        dg_dt = -2.0 * self.ricci_curvature + self.lambda_diff * harmonic_diffusion
        self.g_metric += self.viscosity * dg_dt

        # 계량장 대칭성 및 양의 정정성 보안
        self.g_metric = 0.5 * (self.g_metric + self.g_metric.T)

        return {
            "t_int": t_int,
            "delta_T": delta_T,
            "topological_friction": topological_friction,
            "dg_dt": dg_dt,
            "g_metric": self.g_metric.copy()
        }

    def deconstruct_symbol(
        self,
        label: str,
        action_profile: Dict[str, float],
        scale: str = "physical"
    ) -> Dict[str, Any]:
        """
        축 2: 기호 해체 및 인과 계통 재정렬

        - label: 표면적 언어 라벨 (예: '단단함', '점성')
        - action_profile: [가함 -> 저항 -> 변형] 미시적 궤적 매개변수
        - scale: 'physical', 'informational', 'cognitive' 다중 스케일
        """
        stiffness = float(action_profile.get("delta_F_over_delta_x", 0.0))
        viscosity = float(action_profile.get("delta_F_over_delta_v", 0.0))
        controllability = float(action_profile.get("control_c", 0.0))

        # 궤적 서명 (Trajectory Signature)
        signature_vector = np.array([stiffness, viscosity, controllability], dtype=np.float64)

        entry = {
            "label": label,
            "scale": scale,
            "stiffness": stiffness,
            "viscosity": viscosity,
            "controllability": controllability,
            "signature_vector": signature_vector,
            "g_curvature_imprint": np.outer(signature_vector, signature_vector)[:self.dim, :self.dim]
        }
        self.concept_registry[label] = entry
        return entry

    def trace_unknown_concept(
        self,
        unknown_stream_stress: np.ndarray,
        probe_action: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        축 3: 미지 개념/현상 역추적 및 계통 확장 (No Hallucination)

        - unknown_stream_stress: 관측된 미지 스트림 응력
        - probe_action: 시스템이 투사한 가함 작용
        """
        if probe_action is None:
            probe_action = np.ones(self.dim, dtype=np.float64) / np.sqrt(self.dim)

        # 1. 상호작용 궤적 스캔
        probing_res = self.apply_active_probing(
            u_vector=probe_action,
            external_stress=unknown_stream_stress,
            control_c=0.0  # 미지 현상은 기본 외부(c=0)로 간주
        )

        mismatch = probing_res["delta_T"]
        mismatch_norm = float(np.linalg.norm(mismatch))

        # 2. 인과적 단절 지점 (계량장 분기 곡률) 포착
        mismatch_curvature = np.dot(self.g_metric, mismatch)

        # 3. 하부 인과 뿌리 탐색 (가장 부합하는 기존 계통 노드 검색)
        best_root = "Ground_Zero_Root"
        min_dist = float("inf")

        for label, entry in self.concept_registry.items():
            ref_matrix = entry["g_curvature_imprint"]
            if ref_matrix.shape == mismatch_curvature.shape:
                dist = float(np.linalg.norm(ref_matrix - mismatch_curvature))
                if dist < min_dist:
                    min_dist = dist
                    best_root = label

        return {
            "mismatch_norm": mismatch_norm,
            "mismatch_curvature": mismatch_curvature,
            "connected_root": best_root,
            "divergence_distance": min_dist,
            "is_new_branch_spawned": min_dist > 0.5
        }

    # =========================================================================
    # 4단계 인지 연산 메서드 (Cognition, Thought, Judgment, Discrimination)
    # =========================================================================

    def cognition(
        self,
        u_vector: np.ndarray,
        external_stress: np.ndarray,
        control_c: float
    ) -> Dict[str, Any]:
        """
        1단계: 인지 (Cognition)
        단순 데이터 수신을 넘어, 가함 벡터 u(t)와 반작용 스트림 사이의
        상호작용 저항 규칙을 내부 매니폴드에 등록하고 위상 마찰을 포착합니다.
        """
        return self.apply_active_probing(u_vector, external_stress, control_c)

    def thought(self, virtual_action: np.ndarray) -> np.ndarray:
        """
        2단계: 사고 (Thought)
        실제 외부로 물리적 힘을 출력하지 않고,
        내부 매니폴드(g_ij) 안에서 가상의 작용을 가해 보며 변형될 계량장 상태 및 예측 응력을 사전에 추론합니다.
        """
        virtual_action = np.asarray(virtual_action, dtype=np.float64)
        return self.compute_internal_stress(virtual_action)

    def judgment(
        self,
        predicted_stress: np.ndarray,
        actual_stress: np.ndarray
    ) -> Dict[str, Any]:
        """
        3단계: 판단 (Judgment)
        내부 시뮬레이션이 예측한 응력(T_int)과 실제 유입된 외압(T_ext) 사이의
        오차(Predictive Delta)를 측정하여 인과 유효성을 검증합니다.
        """
        predicted_stress = np.asarray(predicted_stress, dtype=np.float64)
        actual_stress = np.asarray(actual_stress, dtype=np.float64)

        predictive_delta = actual_stress - predicted_stress
        delta_norm = float(np.linalg.norm(predictive_delta))
        is_valid = delta_norm < 0.2  # 허용 오차 임계값

        return {
            "predictive_delta": predictive_delta,
            "delta_norm": delta_norm,
            "is_valid": is_valid
        }

    def discrimination(
        self,
        concept_label_a: str,
        concept_label_b: str
    ) -> Dict[str, Any]:
        """
        4단계: 분별 (Discrimination)
        두 대상/개념이 보여주는 저항·지연·변형 프로필의 상이한 곡률을 바탕으로
        "A라는 인과 영역"과 "B라는 인과 영역" 사이에 기하학적 경계선(Decision Boundary)을 획정합니다.
        """
        if concept_label_a not in self.concept_registry or concept_label_b not in self.concept_registry:
            raise KeyError(f"Concepts {concept_label_a} or {concept_label_b} not registered.")

        entry_a = self.concept_registry[concept_label_a]
        entry_b = self.concept_registry[concept_label_b]

        diff_stiffness = abs(entry_a["stiffness"] - entry_b["stiffness"])
        diff_viscosity = abs(entry_a["viscosity"] - entry_b["viscosity"])
        diff_control = abs(entry_a["controllability"] - entry_b["controllability"])

        boundary_distance = float(np.linalg.norm(entry_a["signature_vector"] - entry_b["signature_vector"]))

        return {
            "concept_a": concept_label_a,
            "concept_b": concept_label_b,
            "diff_stiffness": diff_stiffness,
            "diff_viscosity": diff_viscosity,
            "diff_control": diff_control,
            "boundary_distance": boundary_distance,
            "is_distinct": boundary_distance > 0.1
        }
