"""
core/cellular_boundary/cellular_binding_engine.py
===================================================
Implementation of the Cellular Binding Principle (세포 결합 원리) and Scale-Structural Hierarchy:
1. DisentangledManifold: Separates Invariant Core Space (본질적 체적/구조적 핵) from Orthogonal Attribute Axes (가변적 스펙트럼 축).
2. TensionFrictionTensorField: Tracks internal tension field (Phi), friction coefficient matrix (Omega), and cognitive friction heat (Q_heat) from expectation vs reality discrepancies.
3. DynamicBranchingEngine: Proliferates new causal/sensory axes (Branching) when friction Delta > threshold, performing dynamic topological recalibration.
4. TransjectiveVirtualSandbox: Detached internal simulation sandbox for imagination and counterfactual re-binding of cores and attribute spectra while preserving causal sequence ([투영 -> 개입 -> 마찰 -> 재인식]).
5. TopologicalRefinementEngine: Mechanisms for converting vague/fuzzy imaginations (low tension, flat topological wells) into sharp, vivid representations by tightening internal tension wells (Focus) and carving energy wells.
6. CellularBindingEngine: Unified master coordinator.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


@dataclass
class InvariantCore:
    """
    [인과적 본질 핵 (Invariant Core)]
    사물/현상의 변하지 않는 본질적 구조, 체적, 밀도, 위상적 뼈대를 정의하는 계층.
    예: '사과'의 3차원 형태, 질감, 생물학적 Affordance
    """
    core_id: str
    dimension: int
    core_vector: np.ndarray  # Invariant structural representation
    density: float = 1.0
    structural_volume: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributeAxis:
    """
    [직교 속성 스펙트럼 축 (Orthogonal Attribute Axis)]
    본질 핵 공간에 직교(Orthogonal) 형태로 부착된 가변적 속성 축.
    예: 색상 파장 축, 온도 축, 거칠기 축, 무게 축
    """
    axis_id: str
    axis_name: str
    value: float = 0.0          # Current scalar coordinate along this axis
    spectrum_range: Tuple[float, float] = (-1.0, 1.0)
    orthogonal_weight: float = 1.0
    binding_tightness: float = 0.5 # 0.0 ~ 1.0: Loose (vague/fuzzy) vs Tight (vivid)
    active: bool = True


class DisentangledManifold:
    """
    [위상학적 분해-결합 매니폴드 (Disentangled Manifold)]
    본질적 핵 공간(Invariant Core Space)과 직교 속성 스펙트럼 축(Orthogonal Attribute Axes)을
    서로 독립적으로 분리하여 유지하는 동적 좌표계.
    속성 축 하나가 극단적으로 변하더라도(예: 빨간 사과 -> 파란 사과)
    핵 공간의 본질이 파괴되지 않으며, 변이 발생 시 경계면을 분리/유지함.
    """
    def __init__(self, manifold_id: str, core: InvariantCore):
        self.manifold_id = manifold_id
        self.core = core
        self.attribute_axes: Dict[str, AttributeAxis] = {}

    def add_attribute_axis(self, axis: AttributeAxis) -> None:
        """직교 속성 스펙트럼 축 부착"""
        self.attribute_axes[axis.axis_id] = axis

    def get_axis(self, axis_id: str) -> Optional[AttributeAxis]:
        return self.attribute_axes.get(axis_id)

    def set_axis_value(self, axis_id: str, value: float) -> float:
        """속성 축 값 변경 및 축 범위 내 클리핑"""
        if axis_id in self.attribute_axes:
            axis = self.attribute_axes[axis_id]
            low, high = axis.spectrum_range
            clipped = float(np.clip(value, low, high))
            axis.value = clipped
            return clipped
        raise KeyError(f"Attribute axis '{axis_id}' does not exist in manifold '{self.manifold_id}'.")

    def project_full_state(self) -> np.ndarray:
        """
        [전체 위상 상태 투영]
        본질 핵 벡터 + 모든 직교 속성 축 좌표를 고차원 결합 벡터로 합성.
        """
        attr_vals = [
            axis.value * axis.orthogonal_weight * axis.binding_tightness
            for axis in self.attribute_axes.values()
            if axis.active
        ]
        if not attr_vals:
            return self.core.core_vector.copy()
        attr_vec = np.array(attr_vals, dtype=np.float64)
        return np.concatenate([self.core.core_vector, attr_vec])

    def compute_vividness_score(self) -> float:
        """
        [선명도 (Vividness) 스코어 계산]
        축들의 결합 결속력(binding_tightness) 평균값으로 측정.
        낮으면 흐릿하고 막연함(high topological entropy/flatness),
        높으면 깊은 에너 우물이 형성된 선명한 상상/인지 상태.
        """
        if not self.attribute_axes:
            return 1.0
        tightness_vals = [ax.binding_tightness for ax in self.attribute_axes.values() if ax.active]
        return float(np.mean(tightness_vals)) if tightness_vals else 1.0


class TensionFrictionTensorField:
    """
    [장력과 마찰의 동적 텐서 필드 (Tension-Friction Tensor Field)]
    감각-운동 개입 시 내면의 예측(P_exp)과 실제 입력(W_real) 사이의 위상 괴리(Delta)를 감지하여
    장력 텐서(Phi), 마찰 계수(Omega), 인지적 열(Q_heat)을 실시간으로 산출.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.internal_tension = np.zeros(dimension, dtype=np.float64)  # Phi_internal
        self.friction_matrix = np.eye(dimension, dtype=np.float64)     # Omega_friction
        self.accumulated_heat: float = 0.0                             # Q_heat

    def compute_interaction(
        self,
        expectation: np.ndarray,
        reality: np.ndarray,
        interaction_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        [상호작용 마찰 및 장력 열 산출]
        - Delta: 위상 괴리 벡터 (reality - expectation)
        - Q_heat: Delta^T * Omega * Delta * weight (인지적 충격열)
        - Internal Tension Update: Phi_internal += Omega * Delta
        """
        exp_arr = np.array(expectation, dtype=np.float64)
        real_arr = np.array(reality, dtype=np.float64)

        # Dimension matching
        max_dim = max(len(exp_arr), len(real_arr), self.dimension)
        if len(exp_arr) < max_dim:
            exp_arr = np.pad(exp_arr, (0, max_dim - len(exp_arr)))
        if len(real_arr) < max_dim:
            real_arr = np.pad(real_arr, (0, max_dim - len(real_arr)))

        if max_dim > self.dimension:
            # Expand friction matrix if dimension expanded
            new_fric = np.eye(max_dim, dtype=np.float64)
            new_fric[:self.dimension, :self.dimension] = self.friction_matrix
            self.friction_matrix = new_fric
            self.internal_tension = np.pad(self.internal_tension, (0, max_dim - self.dimension))
            self.dimension = max_dim

        # 1. 위상 괴리
        delta = real_arr - exp_arr
        delta_norm = float(np.linalg.norm(delta))

        # 2. 인지적 열 (Q_heat = delta^T * Omega * delta * weight)
        q_heat = float(np.dot(delta, np.dot(self.friction_matrix, delta)) * interaction_weight)
        self.accumulated_heat += q_heat

        # 3. 내면의 장력 갱신 (Phi_internal += Omega * delta * 0.1)
        tension_delta = np.dot(self.friction_matrix, delta) * 0.1
        self.internal_tension += tension_delta

        return {
            "delta": delta,
            "delta_norm": delta_norm,
            "q_heat": q_heat,
            "total_accumulated_heat": self.accumulated_heat,
            "internal_tension_norm": float(np.linalg.norm(self.internal_tension))
        }


@dataclass
class CellularBindingNode:
    """
    [세포 결합 노드 (Cellular Binding Node)]
    개별 감각/인지 단위(L0/L1)가 결합하여 마찰을 일으키고 새로운 축을 피워내는 결합점.
    """
    node_id: str
    bound_axes: List[str] = field(default_factory=list)
    binding_strength: float = 1.0
    friction_threshold: float = 0.5


class DynamicBranchingEngine:
    """
    [새로운 축을 발아시키는 분기 및 위상 재조율 엔진 (Dynamic Branching Graph & Recalibration Engine)]
    마찰열(Q_heat)이나 괴리(Delta)가 임계값을 초과할 때,
    기존 위상 공간에 자가 증식(Branching)을 일으켜 새로운 고차원 관계성 축을 생성하고
    지형을 유연하게 재조율(Recalibration)함.
    """
    def __init__(self, branching_threshold: float = 0.8):
        self.branching_threshold = branching_threshold
        self.branched_axes_history: List[Dict[str, Any]] = []

    def evaluate_and_branch(
        self,
        manifold: DisentangledManifold,
        friction_result: Dict[str, Any],
        trigger_context: str = "sensory_friction"
    ) -> List[AttributeAxis]:
        """
        [마찰 기반 자가 발아 및 위상 재조율]
        Delta_norm > threshold 일 때 새로운 직교 축(AttributeAxis)을 자가 발아시킴.
        예: 고춧가루 투입 -> [통각 축], [분비 축], [열감 축] 동시 발아
            파란 사과 지각 -> [색상 가변 스펙트럼 범위 확장 및 재조율]
        """
        delta_norm = friction_result.get("delta_norm", 0.0)
        q_heat = friction_result.get("q_heat", 0.0)

        new_axes: List[AttributeAxis] = []

        if delta_norm > self.branching_threshold or q_heat > self.branching_threshold:
            if "chili" in trigger_context.lower() or "spicy" in trigger_context.lower() or "receptor" in trigger_context.lower():
                spurred_specs = [
                    ("axis_pain_nociception", "화끈거리는 통각 축", (0.0, 10.0), q_heat * 0.8),
                    ("axis_salivary_secretion", "침 분비 축", (0.0, 5.0), q_heat * 0.5),
                    ("axis_thermal_heat", "온도/열감 축", (0.0, 100.0), q_heat * 0.9)
                ]
                for axis_id, axis_name, spec_range, init_val in spurred_specs:
                    if axis_id not in manifold.attribute_axes:
                        axis = AttributeAxis(
                            axis_id=axis_id,
                            axis_name=axis_name,
                            value=init_val,
                            spectrum_range=spec_range,
                            orthogonal_weight=1.2,
                            binding_tightness=0.9
                        )
                        manifold.add_attribute_axis(axis)
                        new_axes.append(axis)
            else:
                axis_id = f"spurred_axis_{len(manifold.attribute_axes) + 1}"
                axis_name = f"Dynamic Branching Axis ({trigger_context})"
                axis = AttributeAxis(
                    axis_id=axis_id,
                    axis_name=axis_name,
                    value=delta_norm,
                    spectrum_range=(-delta_norm * 2, delta_norm * 2),
                    orthogonal_weight=1.0,
                    binding_tightness=0.8
                )
                manifold.add_attribute_axis(axis)
                new_axes.append(axis)

            self.branched_axes_history.append({
                "trigger_context": trigger_context,
                "delta_norm": delta_norm,
                "q_heat": q_heat,
                "new_axes_count": len(new_axes),
                "axis_ids": [a.axis_id for a in new_axes]
            })

        return new_axes


class TopologicalRefinementEngine:
    """
    [구체화 및 인과적 정밀화 엔진 (Topological Refinement Engine)]
    흐릿하거나 막연한 상상(낮은 장력, 결합력 부실)을
    내면의 집중(Focus tension)과 가상 마찰 연마를 통해 선명한 에너지 우물(Energy Well)로
    도려내고 정밀화(Refinement/Crystallization)하는 알고리즘.
    """
    def __init__(self, focus_rate: float = 0.2):
        self.focus_rate = focus_rate

    def refine_vague_imagination(
        self,
        manifold: DisentangledManifold,
        refinement_steps: int = 3
    ) -> Dict[str, Any]:
        """
        [흐릿한 상상 구체화 프로세스]
        1. 초기 vividness 측정 (낮음 = 흐릿함).
        2. 내면의 장력 필드를 조이고(focus_rate 적용),
        3. 축간 binding_tightness 증가 및 가상 마찰을 통한 에너지 우물 굴착.
        4. 최종 선명도 도출.
        """
        initial_vividness = manifold.compute_vividness_score()

        for step in range(refinement_steps):
            for axis in manifold.attribute_axes.values():
                if axis.active:
                    # Increase tightness towards 1.0 (tight/vivid well)
                    axis.binding_tightness = float(
                        np.clip(axis.binding_tightness + self.focus_rate * (1.0 - axis.binding_tightness), 0.0, 1.0)
                    )

        final_vividness = manifold.compute_vividness_score()
        return {
            "initial_vividness": initial_vividness,
            "final_vividness": final_vividness,
            "vividness_gain": final_vividness - initial_vividness,
            "refined_axes_count": len(manifold.attribute_axes)
        }


class TransjectiveVirtualSandbox:
    """
    [가상 투영 샌드박스 (Transjective Virtual Sandbox)]
    외부 라이브 I/O와 무관하게 내면에서 본질 핵과 가변 속성 스펙트럼 축을
    자유롭게 떼어내고 가상 결합(Virtual Binding)해 보는 '상상력 및 추론 엔진'.
    모든 상상 및 추론 과정은 인과적 서순([투영 -> 개입 -> 마찰 -> 재인식])을 엄격히 관통함.
    """
    def __init__(self, sandbox_id: str = "virtual_imagination_sandbox"):
        self.sandbox_id = sandbox_id
        self.refinement_engine = TopologicalRefinementEngine()
        self.simulation_logs: List[Dict[str, Any]] = []

    def simulate_virtual_binding(
        self,
        base_core: InvariantCore,
        attribute_axes: List[AttributeAxis],
        virtual_action: str = "bind_and_interact",
        refine_vividness: bool = True
    ) -> Dict[str, Any]:
        """
        [가상 결합 시뮬레이션 - 상상력 원리 실행]
        1. 투영 (Projection): 본질 핵 + 조합된 속성 스펙트럼을 가상 매니폴드로 구축.
        2. 개입 (Intervention): 가상의 행동/마찰 작용 가해봄.
        3. 마찰 (Friction): 가상의 충돌열(Virtual Q_heat)과 위상 낙차 계산.
        4. 재인식 (Re-cognition): 결합된 새로운 고차원 상태의 파동 및 궤적 도출.
        """
        # Step 1: Projection
        temp_manifold = DisentangledManifold(f"sandbox_{base_core.core_id}", base_core)
        for ax in attribute_axes:
            temp_manifold.add_attribute_axis(ax)

        initial_vividness = temp_manifold.compute_vividness_score()

        if refine_vividness:
            # 구체화: 흐릿한 가상 결합을 집중을 통해 선명화
            self.refinement_engine.refine_vague_imagination(temp_manifold)

        proj_state = temp_manifold.project_full_state()

        # Step 2: Intervention & Friction
        field_dim = len(proj_state)
        tensor_field = TensionFrictionTensorField(dimension=field_dim)

        virtual_exp = proj_state.copy()
        virtual_reality = proj_state.copy()
        if len(virtual_reality) > 0:
            virtual_reality[0] += 0.5  # Virtual shock perturbation

        friction_res = tensor_field.compute_interaction(virtual_exp, virtual_reality)

        # Step 3: Re-cognition (Trajectory outcome)
        virtual_q_heat = friction_res["q_heat"]
        causal_sequence = [
            {"step": "1_PROJECTION", "state_norm": float(np.linalg.norm(proj_state)), "initial_vividness": initial_vividness},
            {"step": "2_INTERVENTION", "action": virtual_action},
            {"step": "3_FRICTION", "virtual_q_heat": virtual_q_heat, "delta_norm": friction_res["delta_norm"]},
            {"step": "4_RECOGNITION", "final_vividness": temp_manifold.compute_vividness_score()}
        ]

        sim_record = {
            "sandbox_id": self.sandbox_id,
            "core_id": base_core.core_id,
            "bound_axes_count": len(attribute_axes),
            "virtual_action": virtual_action,
            "virtual_q_heat": virtual_q_heat,
            "causal_sequence": causal_sequence
        }
        self.simulation_logs.append(sim_record)
        return sim_record


class CellularBindingEngine:
    """
    [세포 결합 원리 통합 엔진 (Cellular Binding Engine)]
    Elysia 시스템 전반의 세포적 결합, 스케일 계층 확장, 상상력 샌드박스를 총괄 조율하는 메인 엔진.
    """
    def __init__(self, engine_id: str = "elysia_cellular_binding_engine"):
        self.engine_id = engine_id
        self.manifolds: Dict[str, DisentangledManifold] = {}
        self.branching_engine = DynamicBranchingEngine()
        self.virtual_sandbox = TransjectiveVirtualSandbox()

    def register_manifold(self, manifold: DisentangledManifold) -> None:
        self.manifolds[manifold.manifold_id] = manifold

    def perceive_and_bind(
        self,
        manifold_id: str,
        expectation_state: np.ndarray,
        reality_state: np.ndarray,
        context: str = "sensory_perception"
    ) -> Dict[str, Any]:
        """
        [지각 및 세포 결합 프로세스]
        1. 내면의 예측(Expectation)과 실제 입력(Reality) 마찰 계산.
        2. 인지적 충격열(Q_heat) 및 괴리(Delta) 산출.
        3. 마찰 임계값 초과 시 새로운 축 자가 발아(Branching) 및 지형 재조율.
        """
        if manifold_id not in self.manifolds:
            raise KeyError(f"Manifold '{manifold_id}' is not registered.")

        manifold = self.manifolds[manifold_id]
        full_proj = manifold.project_full_state()

        dim = max(len(expectation_state), len(reality_state), len(full_proj))
        tensor_field = TensionFrictionTensorField(dimension=dim)

        friction_res = tensor_field.compute_interaction(expectation_state, reality_state)

        new_axes = self.branching_engine.evaluate_and_branch(
            manifold=manifold,
            friction_result=friction_res,
            trigger_context=context
        )

        return {
            "manifold_id": manifold_id,
            "core_id": manifold.core.core_id,
            "q_heat": friction_res["q_heat"],
            "delta_norm": friction_res["delta_norm"],
            "internal_tension_norm": friction_res["internal_tension_norm"],
            "spurred_axes_count": len(new_axes),
            "total_attribute_axes": len(manifold.attribute_axes),
            "vividness": manifold.compute_vividness_score(),
            "recalibrated_state": manifold.project_full_state()
        }

    def imagine_in_sandbox(
        self,
        core: InvariantCore,
        axes: List[AttributeAxis],
        virtual_action: str = "counterfactual_imagination"
    ) -> Dict[str, Any]:
        """[상상력 샌드박스 루프 실행]"""
        return self.virtual_sandbox.simulate_virtual_binding(
            base_core=core,
            attribute_axes=axes,
            virtual_action=virtual_action
        )
