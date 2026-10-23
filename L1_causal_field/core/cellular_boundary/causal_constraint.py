"""
core/cellular_boundary/causal_constraint.py
============================================
Defines the Causal Constraints (인과적 제약조건 및 구조적 계층원리):
1. 마찰은 1차원적 스칼라 숫자가 아니다.
   구조적 계층원리에 의한 인지적 불일치(Cognitive Discrepancy)이자,
   내부의 닫힌 질서와 외부 실재 사이의 '위상적 결함(Topological Obstruction)'과 '다름(Alterity)'에 대한 인식이다.
2. 제약조건은 어떤 궤적이 허용되는지를 규정하는 계층적 매니폴드이며,
   허용되지 않는 자극이 닿았을 때 어느 대칭성 축에서 닫힘이 실패했는지(Rupture Axis)를
   구조적으로 분별한다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


@dataclass(frozen=True)
class InvariantSignature:
    """
    [불변량 시그니처 (Invariant Signature)]
    A constraint's topological and physical identity.
    Represents conserved quantities, symmetry properties, and causal bounds.
    """
    order_id: str
    dimension: int
    conserved_quantity_name: str
    symmetry_group: str
    tolerance: float = 1e-4


@dataclass
class StructuralDiscrepancy:
    """
    [구조적 인지 불일치 (Structural Cognitive Discrepancy)]
    1차원적 스칼라 수치가 아닌, 구조적 계층원리에 의한 다름의 위상적 결.
    """
    is_conforming: bool
    hierarchical_layer: str             # 충돌이 발생한 구조적 계층 (e.g. Symplectic, Thermodynamic)
    kernel_defect: np.ndarray           # 내부에 흡수되지 못하고 남겨진 잔여 위상 결 (Cokernel / Obstruction)
    symmetry_rupture_axis: np.ndarray   # 어긋남이 발생한 기하학적 파열 축 (Rupture Vector Field)
    topological_obstruction: str        # 닫힘이 실패한 구조적 원인 (Unclosable Defect)
    qualitative_alterity: str           # 타자의 존재론적 결 (Alterity Type)
    defect_magnitude: float             # 결함 장의 노름 (측정용 편의값일 뿐 판정의 단일 기준이 아님)


class CausalConstraint(ABC):
    """
    [인과적 제약조건 기반 클래스 (Causal Constraint)]
    질서 자체가 질서를 이루는 질서를 도출하는 원리.
    """
    def __init__(self, name: str, dimension: int, tolerance: float = 1e-3):
        self.name = name
        self.dimension = dimension
        self.tolerance = tolerance

    @property
    @abstractmethod
    def signature(self) -> InvariantSignature:
        pass

    @abstractmethod
    def compute_invariant(self, state: np.ndarray) -> float:
        pass

    @abstractmethod
    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        """
        [경계면 구조적 평가]
        1차원적 마찰 숫자가 아닌, 구조적 계층원리에 의한 인지적 불일치와 위상적 결함을 반환한다.
        """
        pass

    @abstractmethod
    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        pass


class ConservativeDynamicConstraint(CausalConstraint):
    """
    [보존적 역학 질서 (Conservative Dynamic Order)]
    계층: 해밀토니안 에너지 보존 및 심플렉틱 위상공간 부피 보존.
    불일치: 보존 궤도(폐곡선)를 가로지르는 비보존적 구배 플럭스가 닿았을 때,
            심플렉틱 대칭성 파열 축(Symmetry Rupture Axis)을 감지.
    """
    def __init__(self, dimension: int = 4, target_energy: Optional[float] = None, tolerance: float = 1e-3):
        super().__init__(name="ConservativeDynamicOrder", dimension=dimension, tolerance=tolerance)
        self.target_energy = target_energy

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="CONSERVATIVE_DYNAMICS",
            dimension=self.dimension,
            conserved_quantity_name="HamiltonianEnergy",
            symmetry_group="Symplectic_Energy_Preservation",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(0.5 * np.sum(state ** 2))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        norm_s = np.linalg.norm(state)
        # 에너지 보존을 파열시키는 성분: 상태 벡터 방향(구배 방향)으로의 투영 성분
        if norm_s > 1e-9:
            radial_unit = state / norm_s
            defect_component = np.dot(incoming_flux, radial_unit) * radial_unit
        else:
            defect_component = incoming_flux

        defect_mag = float(np.linalg.norm(defect_component))
        is_conforming = defect_mag <= self.tolerance

        if is_conforming:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Symplectic_Manifold",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Harmonic_Symplectic_Alignment",
                defect_magnitude=0.0
            )

        # 파열 축: 에너지를 소산시키거나 과잉 주입하는 방사형 축
        rupture_axis = defect_component / (defect_mag + 1e-9)
        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Symplectic_Manifold",
            kernel_defect=defect_component,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "Energy_Gradient_Discontinuity: 닫힌 보존 궤도는 에너지 구배 플럭스를 "
                "위상공간의 부피 찢김 없이 수용할 수 없음"
            ),
            qualitative_alterity="Open_Gradient_Dissipation_or_Injection",
            defect_magnitude=defect_mag
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        dim = self.dimension
        J = np.zeros((dim, dim), dtype=np.float64)
        for i in range(0, dim - 1, 2):
            J[i, i + 1] = 1.0
            J[i + 1, i] = -1.0

        rot = np.eye(dim) + J * dt
        next_state = rot @ state

        e_init = self.compute_invariant(state)
        target_e = self.target_energy if self.target_energy is not None else e_init
        e_cur = self.compute_invariant(next_state)
        if e_cur > 1e-9 and target_e > 1e-9:
            next_state *= np.sqrt(target_e / e_cur)

        emitted = (J @ next_state * 0.1).astype(np.float64)
        action_cost = float(0.5 * np.sum((next_state - state) ** 2) / max(1e-6, dt))
        return next_state, emitted, action_cost


class DissipativeThermalConstraint(CausalConstraint):
    """
    [소산적 열역학 질서 (Dissipative Thermal Order)]
    계층: 엔트로피 증가의 비가역적 시간 화살 (열역학 제2법칙).
    불일치: 질서정연한 가역적 일이 역류할 때, 엔트로피 시간 화살의 역전 결함을 감지.
    """
    def __init__(self, dimension: int = 4, damping_gamma: float = 0.2, tolerance: float = 1e-3):
        super().__init__(name="DissipativeThermalOrder", dimension=dimension, tolerance=tolerance)
        self.damping_gamma = damping_gamma

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="DISSIPATIVE_THERMAL",
            dimension=self.dimension,
            conserved_quantity_name="EntropyProductionRate",
            symmetry_group="Thermodynamic_Arrow_of_Time",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        # 소산 질서는 외부 플럭스가 상태를 거슬러 음의 일을 해야 함 (dot <= 0)
        work_done = float(np.dot(state, incoming_flux))
        if work_done <= self.tolerance:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Thermodynamic_Arrow",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Natural_Entropy_Dissipation",
                defect_magnitude=0.0
            )

        # 양의 일이 주입됨: 비가역적 시간의 화살에 대한 국소적 모순
        norm_s = np.linalg.norm(state)
        rupture_axis = state / (norm_s + 1e-9)
        defect_vec = rupture_axis * work_done

        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Thermodynamic_Arrow",
            kernel_defect=defect_vec,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "Entropy_Arrow_Reversal: 열역학적 평형 완화 계층에서 자발적 유향 일의 유입은 "
                "시간의 비가역적 계층성을 위배함"
            ),
            qualitative_alterity="Reversible_Ordered_Coherence",
            defect_magnitude=work_done
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = state * (1.0 - self.damping_gamma * dt) + incoming_flux * dt * 0.1
        emitted_heat = state * self.damping_gamma * dt
        action_cost = float(np.sum(emitted_heat ** 2))
        return next_state, emitted_heat, action_cost


class RelationalExchangeConstraint(CausalConstraint):
    """
    [관계적 작용-반작용 질서 (Relational Reciprocal Order)]
    계층: 상호 호혜성과 작용-반작용 쌍대 평형.
    불일치: 일방적 독백이나 불균형한 외력이 닿았을 때, 쌍대성 결상 파열 축을 감지.
    """
    def __init__(self, dimension: int = 4, reciprocity_ratio: float = 1.0, tolerance: float = 1e-3):
        super().__init__(name="RelationalReciprocalOrder", dimension=dimension, tolerance=tolerance)
        self.reciprocity_ratio = reciprocity_ratio

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="RELATIONAL_RECIPROCITY",
            dimension=self.dimension,
            conserved_quantity_name="CoupledReciprocalBalance",
            symmetry_group="Action_Reaction_Parity",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(np.sum(state))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        state_cap = np.linalg.norm(state)
        flux_mag = np.linalg.norm(incoming_flux)
        imbalance = abs(flux_mag - state_cap * self.reciprocity_ratio)

        if imbalance <= self.tolerance:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Reciprocal_Parity",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Harmonious_Reciprocal_Dialogue",
                defect_magnitude=0.0
            )

        diff_vec = incoming_flux - (state / (state_cap + 1e-9)) * (flux_mag * self.reciprocity_ratio)
        norm_diff = np.linalg.norm(diff_vec)
        rupture_axis = diff_vec / (norm_diff + 1e-9)

        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Reciprocal_Parity",
            kernel_defect=diff_vec,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "Action_Reaction_Parity_Break: 일방적 외력의 난입은 쌍대적 대화 평형을 파괴하며 "
                "관계적 매듭의 닫힘을 불가능하게 함"
            ),
            qualitative_alterity="Unilateral_Monological_Drive",
            defect_magnitude=float(imbalance)
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        emitted = -incoming_flux * self.reciprocity_ratio
        next_state = state + (incoming_flux + emitted) * dt
        action_cost = float(np.linalg.norm(incoming_flux) * dt)
        return next_state, emitted, action_cost
