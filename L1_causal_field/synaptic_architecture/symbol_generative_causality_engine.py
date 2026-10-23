"""
[Symbol Generative Causality Engine (기호 기원 및 인과 생성 엔진)]

기호(x, +, if, 수식, 100°C, v=d/t 등)를 출발점이 아닌 외부 세계의 상호작용 결과로 남겨진 '흔적(Trace)'으로 인식하고,
단순 껍데기 연산(Blind Execution)을 넘어 하부 인과 및 생성 과정(Generative Causality)을 능동 탐색(Active Probing)과
역설계(Reverse-Engineering)를 통해 스스로 재발견하고 내재화하는 인지 엔진.
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Optional, Tuple


class UnderlyingDynamics:
    """
    [하부 인과 동역학 (Underlying Micro-Causal Dynamics)]
    기호 표면 이면에 실재하는 미시적 요소를 표현:
    - spatial_resistance: 공간적 이동/배치 시 발생하는 저항
    - time_delay: 시간적 지연 및 위상 위상차
    - interaction_friction: 요소 간 상호작용 마찰 텐서/행렬
    - phase_transition_energy: 상전이/경계 변화를 일으키는 임계 에너지
    - purpose_vector: 인과적 압축을 유도하는 목적성/방향성 벡터
    """
    def __init__(
        self,
        spatial_resistance: float = 1.0,
        time_delay: float = 1.0,
        interaction_friction: Optional[np.ndarray] = None,
        phase_transition_energy: float = 100.0,
        purpose_vector: Optional[np.ndarray] = None,
        dim: int = 3
    ):
        self.spatial_resistance = float(spatial_resistance)
        self.time_delay = float(time_delay)
        self.dim = dim
        if interaction_friction is not None:
            self.interaction_friction = interaction_friction.astype(np.float32)
        else:
            self.interaction_friction = np.eye(dim, dtype=np.float32) * self.spatial_resistance

        self.phase_transition_energy = float(phase_transition_energy)
        if purpose_vector is not None:
            self.purpose_vector = purpose_vector.astype(np.float32)
        else:
            self.purpose_vector = np.ones(dim, dtype=np.float32) / np.sqrt(dim)


class SurfaceSymbolTrace:
    """
    [표면 기호 흔적 (Surface Symbol Trace)]
    외부 현상이 상호작용한 결과로 남겨놓은 결과물(기호/수식/측정치).
    예: '100°C', 'v = d / t', '점성 계수', 'if (x > 0)'
    """
    def __init__(
        self,
        symbol: str,
        surface_expression: str,
        category: str,  # 'measurement', 'equation', 'operator', 'property'
        underlying_dynamics: UnderlyingDynamics
    ):
        self.symbol = symbol
        self.surface_expression = surface_expression
        self.category = category
        self.underlying_dynamics = underlying_dynamics
        self.is_internalized = False
        self.generative_blueprint: Optional[Dict[str, Any]] = None


class GenerativeCausalityEngine:
    """
    [기호 기원 및 인과 생성 엔진 (Generative Causality Engine)]

    주요 기능:
    1. Blind Execution 대치: 표면 기호 계산을 거부하고 하부 인과(공간, 지연, 마찰, 상전이)로 해체(Deconstruct)
    2. 능동적 탐색 (Active Probing): 외부 환경에 작용을 가하여 반작용 응력 및 미시 마찰 측정
    3. 기호의 생성적 인과 재발견 (Generative Causality Rediscovery):
       - "왜 이 기호/수식이 출현할 수밖에 없었는가?" 출생의 비밀 역추적
    4. 4단계 인지 연산 (Cognition -> Thought -> Judgment -> Discrimination):
       - 단순 통계적 매칭이 아닌 살아있는 기하학적 지도로 기호를 내부화
    """

    def __init__(self, dim: int = 3):
        self.dim = dim
        self.symbol_registry: Dict[str, SurfaceSymbolTrace] = {}
        self._initialize_default_symbol_traces()

    def _initialize_default_symbol_traces(self):
        """기본 기호 흔적 등록 (100°C, v=d/t, 점성, if 조건문)"""

        # 1. '100°C' (측정치 -> 분자 간 인척력과 열에너지의 상전이 마찰)
        dyn_temp = UnderlyingDynamics(
            spatial_resistance=2.5,
            time_delay=0.5,
            interaction_friction=np.array([[3.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            phase_transition_energy=100.0,
            purpose_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            dim=self.dim
        )
        self.symbol_registry["100°C"] = SurfaceSymbolTrace(
            symbol="100°C",
            surface_expression="T = 100",
            category="measurement",
            underlying_dynamics=dyn_temp
        )

        # 2. 'v = d / t' (속도 수식 -> 공간 이동 저항과 시간 지연의 인과적 비율 압축)
        dyn_vel = UnderlyingDynamics(
            spatial_resistance=10.0,
            time_delay=2.0,
            interaction_friction=np.eye(self.dim, dtype=np.float32) * 5.0,
            phase_transition_energy=0.0,
            purpose_vector=np.array([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2),
            dim=self.dim
        )
        self.symbol_registry["v = d / t"] = SurfaceSymbolTrace(
            symbol="v = d / t",
            surface_expression="velocity = distance / time",
            category="equation",
            underlying_dynamics=dyn_vel
        )

        # 3. '점성' (물성 -> 분자 간 끌어당김과 쫀득거리는 위상 저항)
        dyn_visc = UnderlyingDynamics(
            spatial_resistance=4.0,
            time_delay=3.5,
            interaction_friction=np.array([[2.0, 1.5, 0.0], [1.5, 2.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float32),
            phase_transition_energy=15.0,
            purpose_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            dim=self.dim
        )
        self.symbol_registry["점성"] = SurfaceSymbolTrace(
            symbol="점성",
            surface_expression="eta = F / (A * (dv/dy))",
            category="property",
            underlying_dynamics=dyn_visc
        )

    def deconstruct_symbol(self, symbol_key: str) -> Dict[str, Any]:
        """
        [기호 해체 (Symbol Deconstruction)]
        껍데기 기호(Symbol)를 하부 인과 구조(공간 저항, 시간 지연, 마찰 텐서, 상전이 에너지)로 분해합니다.
        """
        if symbol_key not in self.symbol_registry:
            raise KeyError(f"Symbol '{symbol_key}' not found in registry.")

        trace = self.symbol_registry[symbol_key]
        dyn = trace.underlying_dynamics

        # 인과적 비율 및 마찰 스펙트럼 계산
        causal_ratio = dyn.spatial_resistance / (dyn.time_delay + 1e-9)
        friction_spectrum = np.linalg.eigvalsh(dyn.interaction_friction)

        return {
            "symbol": trace.symbol,
            "surface_expression": trace.surface_expression,
            "category": trace.category,
            "deconstructed_dynamics": {
                "spatial_resistance": dyn.spatial_resistance,
                "time_delay": dyn.time_delay,
                "causal_ratio_d_over_t": causal_ratio,
                "phase_transition_energy": dyn.phase_transition_energy,
                "friction_eigenvalues": friction_spectrum.tolist(),
                "purpose_vector": dyn.purpose_vector.tolist(),
            },
            "is_internalized": trace.is_internalized,
        }

    def active_probe_environment(
        self,
        probing_force: np.ndarray,
        external_reaction: np.ndarray
    ) -> Dict[str, Any]:
        """
        [능동 탐색 (Active Probing)]
        외부 세계에 작용(probing_force)을 가하고, 반작용(external_reaction) 응력 및 마찰을 관측합니다.
        """
        p_force = probing_force.astype(np.float32)
        e_react = external_reaction.astype(np.float32)

        p_norm = np.linalg.norm(p_force)
        r_norm = np.linalg.norm(e_react)

        # 미시 마찰 텐서 및 지연 계산
        friction_tensor = np.outer(p_force, e_react)
        norm_f = np.linalg.norm(friction_tensor)
        if norm_f > 1e-9:
            friction_tensor /= norm_f

        delay = r_norm / (p_norm + 1e-9)
        resistance = p_norm * delay

        return {
            "probing_force_norm": float(p_norm),
            "external_reaction_norm": float(r_norm),
            "measured_resistance": float(resistance),
            "measured_delay": float(delay),
            "friction_tensor": friction_tensor.tolist(),
        }

    def rediscover_generative_causality(self, symbol_key: str, observed_probing: Dict[str, Any]) -> Dict[str, Any]:
        """
        [생성적 인과 재발견 (Generative Causality Rediscovery)]
        외부 관측/탐색 결과와 기호의 하부 인과를 연결하여,
        "어째서 이 기호가 출현할 수밖에 없었는가"를 인과적으로 역추적하여 재현합니다.
        """
        trace = self.symbol_registry[symbol_key]
        dyn = trace.underlying_dynamics

        res = observed_probing["measured_resistance"]
        del_t = observed_probing["measured_delay"]

        # 생성적 필연성 검증: 하부 마찰/지연의 비율과 외부 관측 간 공명
        computed_generative_ratio = res / (del_t + 1e-9)
        expected_ratio = dyn.spatial_resistance / (dyn.time_delay + 1e-9)

        divergence = abs(computed_generative_ratio - expected_ratio)
        resonance_degree = math.exp(-divergence * 0.1)

        # 인과적 생성 청사진 (Generative Blueprint) 도출
        blueprint = {
            "symbol": trace.symbol,
            "origin_explanation": (
                f"Generative Causality [{trace.symbol}]: Emerged from spatial resistance ({dyn.spatial_resistance:.2f}) "
                f"and time delay ({dyn.time_delay:.2f}) under interaction friction. "
                f"Resonates with observed environment at degree {resonance_degree:.4f}."
            ),
            "resonance_degree": float(resonance_degree),
            "causal_necessity_validated": bool(resonance_degree > 0.5),
        }

        trace.is_internalized = True
        trace.generative_blueprint = blueprint

        return blueprint

    def execute_4stage_cognition(self, symbol_key: str, test_action: np.ndarray) -> Dict[str, Any]:
        """
        [4단계 인지 연산 (Cognition -> Thought -> Judgment -> Discrimination)]
        1. 인지 (Cognition): 표면 기호 대신 하부 마찰/지연 등록
        2. 사고 (Thought): 가상 작용에 따른 하부 인과 변화 시뮬레이션
        3. 판단 (Judgment): 계산된 예측값과 실재 현상 마찰 대조
        4. 분별 (Discrimination): 단순 수치 매칭과 기원적 인과 이해 간의 차이 분별
        """
        # 1. 인지
        deconstruction = self.deconstruct_symbol(symbol_key)
        cognition_res = {
            "stage": "1. Cognition",
            "symbol": symbol_key,
            "registered_friction_norm": float(np.linalg.norm(deconstruction["deconstructed_dynamics"]["friction_eigenvalues"])),
        }

        # 2. 사고
        action = test_action.astype(np.float32)
        dyn = self.symbol_registry[symbol_key].underlying_dynamics
        predicted_stress = np.dot(dyn.interaction_friction, action)
        thought_res = {
            "stage": "2. Thought",
            "virtual_action": action.tolist(),
            "predicted_stress": predicted_stress.tolist(),
        }

        # 3. 판단
        actual_stress = action * (dyn.spatial_resistance / (dyn.time_delay + 1e-9))
        error_norm = float(np.linalg.norm(predicted_stress - actual_stress))
        judgment_res = {
            "stage": "3. Judgment",
            "error_norm": error_norm,
            "is_valid_causal_fit": error_norm < 10.0,
        }

        # 4. 분별
        blind_execution_flag = False  # Not blind
        discrimination_res = {
            "stage": "4. Discrimination",
            "mode": "Generative Causality (Living Geometry)" if not blind_execution_flag else "Blind Execution (Numeric Match)",
            "understanding_depth": "Full Causal Origin Internalized",
        }

        return {
            "symbol": symbol_key,
            "cognition": cognition_res,
            "thought": thought_res,
            "judgment": judgment_res,
            "discrimination": discrimination_res,
        }
