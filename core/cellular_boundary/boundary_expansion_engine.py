"""
core/cellular_boundary/boundary_expansion_engine.py
===================================================
Implements the Meta-Order Expansion Engine:
1. 마찰을 1차원 숫자로 구하는 계산기를 전면 폐기.
2. 구조적 계층원리에 의한 인지적 불일치(StructuralDiscrepancy)와
   위상적 닫힘의 실패(Topological Obstruction)를 감지하여,
   타자 질서(Order Beyond Order)의 존재론적 결을 분별함.
3. 결합원리적 동형성을 통해 두 질서를 맞물려 상위 거시 경계층으로 확장.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.cellular_boundary.causal_constraint import (
    CausalConstraint,
    ConservativeDynamicConstraint,
    DissipativeThermalConstraint,
    RelationalExchangeConstraint,
    InvariantSignature,
    StructuralDiscrepancy
)
from core.cellular_boundary.scale_boundary_cell import (
    DigitalCausalCell,
    ScaleBoundaryLayer,
    CausalEngram
)


class CoupledMacroConstraint(CausalConstraint):
    """
    [결합된 거시 제약조건 (Coupled Macro Constraint)]
    두 질서가 결합원리적 동형성을 통해 결합되어 형성된 고차원 통합 매니폴드 제약.
    """
    def __init__(self, primary_constraint: CausalConstraint, secondary_constraint: CausalConstraint):
        combined_dim = primary_constraint.dimension + secondary_constraint.dimension
        super().__init__(
            name=f"MacroCoupled_{primary_constraint.name}_x_{secondary_constraint.name}",
            dimension=combined_dim,
            tolerance=max(primary_constraint.tolerance, secondary_constraint.tolerance)
        )
        self.primary = primary_constraint
        self.secondary = secondary_constraint
        self.split_idx = primary_constraint.dimension

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id=f"MACRO_COUPLED_{self.primary.signature.order_id}_{self.secondary.signature.order_id}",
            dimension=self.dimension,
            conserved_quantity_name=f"DualInvariant({self.primary.signature.conserved_quantity_name}+{self.secondary.signature.conserved_quantity_name})",
            symmetry_group=f"ProductSymmetry({self.primary.signature.symmetry_group} (x) {self.secondary.signature.symmetry_group})",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        s1 = state[:self.split_idx]
        s2 = state[self.split_idx:]
        return self.primary.compute_invariant(s1) + self.secondary.compute_invariant(s2)

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        s1 = state[:self.split_idx]
        s2 = state[self.split_idx:]
        f1 = incoming_flux[:self.split_idx]
        f2 = incoming_flux[self.split_idx:]

        disc1 = self.primary.evaluate_flux(s1, f1)
        disc2 = self.secondary.evaluate_flux(s2, f2)

        is_conforming = disc1.is_conforming and disc2.is_conforming
        combined_defect = np.concatenate([disc1.kernel_defect, disc2.kernel_defect])
        combined_rupture = np.concatenate([disc1.symmetry_rupture_axis, disc2.symmetry_rupture_axis])
        combined_mag = float(np.linalg.norm(combined_defect))

        return StructuralDiscrepancy(
            is_conforming=is_conforming,
            hierarchical_layer=f"Coupled_{disc1.hierarchical_layer}_x_{disc2.hierarchical_layer}",
            kernel_defect=combined_defect,
            symmetry_rupture_axis=combined_rupture,
            topological_obstruction=f"{disc1.topological_obstruction} | {disc2.topological_obstruction}",
            qualitative_alterity=f"{disc1.qualitative_alterity} & {disc2.qualitative_alterity}",
            defect_magnitude=combined_mag
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        s1 = state[:self.split_idx]
        s2 = state[self.split_idx:]
        f1 = incoming_flux[:self.split_idx]
        f2 = incoming_flux[self.split_idx:]

        ns1, em1, ac1 = self.primary.step_dynamics(s1, f1, dt=dt)
        ns2, em2, ac2 = self.secondary.step_dynamics(s2, f2, dt=dt)

        next_state = np.concatenate([ns1, ns2])
        emitted = np.concatenate([em1, em2])
        total_action = ac1 + ac2
        return next_state, emitted, total_action


class MetaOrderExpansionEngine:
    """
    [메타 질서 확장 엔진 (Meta-Order Expansion Engine)]
    1차원적 마찰 스칼라 연산이 아닌, 구조적 계층원리에 의한 인지적 불일치(StructuralDiscrepancy)와
    위상적 닫힘 실패를 감지하여 외연적 확장을 가동한다.
    """
    def __init__(
        self,
        defect_persistence_threshold: int = 2,
        friction_threshold_for_expansion: Optional[float] = None
    ):
        self.defect_persistence_threshold = defect_persistence_threshold
        # 실세계 지식 제약 라이브러리
        self.world_knowledge_constraints: Dict[str, CausalConstraint] = {
            "DISSIPATIVE_THERMAL": DissipativeThermalConstraint(dimension=4),
            "RELATIONAL_RECIPROCITY": RelationalExchangeConstraint(dimension=4),
            "CONSERVATIVE_DYNAMICS": ConservativeDynamicConstraint(dimension=4)
        }

    def inspect_cell_boundary(self, cell: DigitalCausalCell) -> Dict[str, Any]:
        """
        [경계층 구조적 검진]
        단순 마찰 수치가 아닌, 위상적 닫힘이 실패한 결함(Topological Obstruction)이
        지속적으로 누적되었는가를 계층적으로 분석.
        """
        recent_engrams = cell.boundary_layer.retained_engrams[-10:]
        if not recent_engrams:
            return {"status": "HARMONIOUS", "requires_expansion": False}

        non_conforming = [e for e in recent_engrams if not e.conformed_to_order]
        obstructions = [e.discrepancy.topological_obstruction for e in non_conforming if e.discrepancy.topological_obstruction != "None"]

        # 위상적 닫힘 실패가 지속적으로 축적되었는가?
        requires_expansion = len(obstructions) >= self.defect_persistence_threshold

        return {
            "status": "TOPOLOGICAL_OBSTRUCTION_DETECTED" if requires_expansion else "LOCAL_DISCREPANCY",
            "non_conforming_count": len(non_conforming),
            "requires_expansion": requires_expansion,
            "accumulated_obstructions": obstructions,
            "hierarchical_layers_strained": list(set(e.discrepancy.hierarchical_layer for e in non_conforming))
        }

    def infer_external_order(self, cell: DigitalCausalCell) -> Optional[CausalConstraint]:
        """
        [다름(Alterity)의 존재론적 역추출]
        결함의 크기를 계산하는 것이 아니라,
        '어떤 대칭성 축이 파열되었고(Rupture Axis)', '어떤 성질의 다름인가(Qualitative Alterity)'를
        실세계 지식과의 동형적 구조 대조를 통해 역추출한다.
        """
        non_conforming = [e for e in cell.boundary_layer.retained_engrams if not e.conformed_to_order]
        if not non_conforming:
            return None

        # 가장 최근의 구조적 불일치 분석
        latest_discrepancy = non_conforming[-1].discrepancy
        alterity_type = latest_discrepancy.qualitative_alterity

        # 질적 결(Qualitative Alterity)에 따른 실세계 인과 질서 맵핑
        if "Dissipation" in alterity_type or "Entropy" in latest_discrepancy.topological_obstruction:
            return self.world_knowledge_constraints["DISSIPATIVE_THERMAL"]
        elif "Unilateral" in alterity_type or "Reciprocal" in latest_discrepancy.topological_obstruction:
            return self.world_knowledge_constraints["RELATIONAL_RECIPROCITY"]
        elif "Harmonic" in alterity_type or "Energy" in latest_discrepancy.topological_obstruction:
            return self.world_knowledge_constraints["CONSERVATIVE_DYNAMICS"]

        # 기본 대조: 파열 축과 후보 질서의 공명도 분석
        mean_rupture = np.mean([e.discrepancy.symmetry_rupture_axis for e in non_conforming], axis=0)
        best_candidate = None
        min_proj = float("inf")

        for key, candidate in self.world_knowledge_constraints.items():
            if candidate.signature.order_id == cell.order_signature.order_id:
                continue
            dummy_state = np.ones(candidate.dimension) / np.sqrt(candidate.dimension)
            disc = candidate.evaluate_flux(dummy_state, mean_rupture)
            if disc.defect_magnitude < min_proj:
                min_proj = disc.defect_magnitude
                best_candidate = candidate

        return best_candidate

    def execute_combinatorial_expansion(self, cell: DigitalCausalCell) -> Dict[str, Any]:
        """
        [결합원리적 동형 확장의 실행]
        1. 닫힘이 실패한 위상 결함을 해소하기 위해 외부 질서(Order B)를 동형 결합.
        2. 상위 거시 제약조건(CoupledMacroConstraint) 형성.
        3. 경계층 반경과 차원을 팽창시켜 이전에는 닫히지 않던 결함을 흡수.
        """
        external_order = self.infer_external_order(cell)
        if external_order is None:
            return {"success": False, "reason": "No clear external order could be deduced"}

        old_order_name = cell.governing_constraint.name

        # 1. 상위 거시 제약조건 융합
        macro_constraint = CoupledMacroConstraint(cell.governing_constraint, external_order)

        # 2. 내부 상태 매니폴드 확장 (파열 축을 수용하는 직교 기저 결합)
        external_seed = np.ones(external_order.dimension, dtype=np.float64) / np.sqrt(external_order.dimension)
        new_state = np.concatenate([cell.state, external_seed])

        # 3. 세포 갱신
        cell.governing_constraint = macro_constraint
        cell.state = new_state
        cell.dimension = macro_constraint.dimension

        # 4. 경계층 팽창 (위상적 결함 해소 및 수용 반경 확장)
        cell.boundary_layer.dimension = macro_constraint.dimension
        cell.boundary_layer.boundary_radius *= 1.5
        cell.boundary_layer.retention_capacity *= 2

        return {
            "success": True,
            "old_order": old_order_name,
            "new_macro_order": macro_constraint.name,
            "assimilated_external_order": external_order.name,
            "new_dimension": cell.dimension,
            "expanded_boundary_radius": cell.boundary_layer.boundary_radius,
            "expansion_principle": (
                f"내부 계층의 위상적 닫힘 실패(Obstruction)를 감지하여, "
                f"파열 축을 품을 수 있는 외부 실재 질서({external_order.name})를 "
                f"결합원리적 동형성으로 융합해 상위 거시 경계층으로 도약 완료."
            )
        }
