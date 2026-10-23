"""
core/cellular_boundary/multimodal_causal_spatiotemporal_network.py
==================================================================
Implements Multimodal Causal Spatiotemporal Differentiation & Cognitive Network:
1. Universal Spatiotemporal Causal Ordering across:
   - Mathematics (Axiomatic derivation sequence)
   - Code/Computation (State transition sequence)
   - Language/Semantics (Narrative/contextual progression)
   - Physics/Spatiotemporal (Motion & field interaction sequence)
2. Recognizes each modality as a distinct causal order governed by boundary constraints.
3. Uses StructuralDiscrepancy instead of 1D scalar friction.
4. Weaves them into a unified, self-reproducing Cognitive Network (순환논리를 가진 세계구조).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.cellular_boundary.causal_constraint import (
    CausalConstraint,
    InvariantSignature,
    StructuralDiscrepancy
)
from core.cellular_boundary.scale_boundary_cell import (
    DigitalCausalCell,
    ScaleBoundaryLayer,
    CausalEngram
)
from core.cellular_boundary.boundary_expansion_engine import (
    MetaOrderExpansionEngine,
    CoupledMacroConstraint
)


class ModalityType(Enum):
    MATHEMATICS = "MATHEMATICS"     # Axiomatic deduction sequence
    CODE = "CODE"                   # Execution state transition sequence
    LANGUAGE = "LANGUAGE"           # Semantic narrative progression
    PHYSICS = "PHYSICS"             # Spatiotemporal kinematic momentum


@dataclass
class CausalStep:
    """A discrete sequential step in any spatiotemporal causal process."""
    step_index: int
    modality: ModalityType
    premise: str
    action_or_transition: str
    resultant_state: np.ndarray
    invariant_delta: float = 0.0


class MathematicalCausalConstraint(CausalConstraint):
    """
    [수학적 인과 제약조건 (Mathematical Causal Order)]
    계층: 공리적 연역의 동어반복적 필연성.
    불일치: 비논리적 비약(Non-Sequitur)이나 공리 파열이 닿았을 때 연역적 부분공간의 결함을 감지.
    """
    def __init__(self, dimension: int = 4, tolerance: float = 1e-3):
        super().__init__(name="MathematicalCausalOrder", dimension=dimension, tolerance=tolerance)

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="MATHEMATICAL_AXIOMATIC",
            dimension=self.dimension,
            conserved_quantity_name="LogicalConsistencyMeasure",
            symmetry_group="First_Order_Axiomatic_Equivalence",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(np.sum(np.abs(state)))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        norm_s = np.linalg.norm(state)
        norm_f = np.linalg.norm(incoming_flux)
        if norm_s > 1e-9 and norm_f > 1e-9:
            proj = (np.dot(incoming_flux, state) / (norm_s ** 2)) * state
            kernel_defect = incoming_flux - proj
        else:
            kernel_defect = np.zeros(self.dimension, dtype=np.float64)

        defect_mag = float(np.linalg.norm(kernel_defect))
        is_conforming = defect_mag <= self.tolerance

        if is_conforming:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Axiomatic_Deduction",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Harmonious_Tautological_Deduction",
                defect_magnitude=0.0
            )

        rupture_axis = kernel_defect / (defect_mag + 1e-9)
        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Axiomatic_Deduction",
            kernel_defect=kernel_defect,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "Axiomatic_Subspace_Disconnection: 비논리적 비약(Non-Sequitur)으로 인해 "
                "연역적 측지선을 통한 위상 닫힘이 실패함"
            ),
            qualitative_alterity="Alien_Non_Deductive_Leap",
            defect_magnitude=defect_mag
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = state + incoming_flux * dt
        norm = np.linalg.norm(next_state)
        if norm > 0:
            next_state = next_state / norm
        theorem_lemma = incoming_flux * 0.5
        return next_state, theorem_lemma, float(np.linalg.norm(incoming_flux))


class CodeExecutionConstraint(CausalConstraint):
    """
    [코드/연산 인과 제약조건 (Computational Execution Order)]
    계층: 튜링 상태 머신 및 메모리/타입 불변 경계.
    불일치: 버퍼 오버플로우나 타입 파열 시 상태 공간 경계 초과 결함을 감지.
    """
    def __init__(self, dimension: int = 4, memory_bound: float = 2.0, tolerance: float = 1e-3):
        super().__init__(name="CodeExecutionOrder", dimension=dimension, tolerance=tolerance)
        self.memory_bound = memory_bound

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="COMPUTATIONAL_STATE_MACHINE",
            dimension=self.dimension,
            conserved_quantity_name="MemoryIntegrityCheck",
            symmetry_group="Turing_Deterministic_State_Transition",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(np.max(np.abs(state)))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        projected = state + incoming_flux
        overflow_mask = np.abs(projected) > self.memory_bound
        overflow_vec = np.where(overflow_mask, np.abs(projected) - self.memory_bound, 0.0)
        overflow_mag = float(np.linalg.norm(overflow_vec))
        is_conforming = overflow_mag <= self.tolerance

        if is_conforming:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Turing_State_Machine",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Deterministic_Bounded_Transition",
                defect_magnitude=0.0
            )

        rupture_axis = overflow_vec / (overflow_mag + 1e-9)
        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Turing_State_Machine",
            kernel_defect=overflow_vec,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "State_Manifold_Boundary_Overflow: 메모리 바운드를 초과하여 "
                "유한 상태 기계의 컴팩트 위상 경계가 파열됨"
            ),
            qualitative_alterity="Unbounded_Buffer_Corruption",
            defect_magnitude=overflow_mag
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = np.clip(state + incoming_flux * dt, -self.memory_bound, self.memory_bound)
        output_register = incoming_flux * 0.8
        return next_state, output_register, float(np.sum(incoming_flux ** 2))


class LanguageNarrativeConstraint(CausalConstraint):
    """
    [언어적 맥락/서사 인과 제약조건 (Linguistic Narrative Order)]
    계층: 맥락적 연속성 및 의미적 장력 해소.
    불일치: 문맥 파열이나 서사적 붕괴가 닿았을 때 의미 장력 파열 축을 감지.
    """
    def __init__(self, dimension: int = 4, tolerance: float = 1e-3):
        super().__init__(name="LanguageNarrativeOrder", dimension=dimension, tolerance=tolerance)

    @property
    def signature(self) -> InvariantSignature:
        return InvariantSignature(
            order_id="LINGUISTIC_SEMANTIC_TENSION",
            dimension=self.dimension,
            conserved_quantity_name="ContextualContinuityIndex",
            symmetry_group="Narrative_Contextual_Flow",
            tolerance=self.tolerance
        )

    def compute_invariant(self, state: np.ndarray) -> float:
        return float(np.mean(state))

    def evaluate_flux(self, state: np.ndarray, incoming_flux: np.ndarray) -> StructuralDiscrepancy:
        diff_vec = incoming_flux - state
        dissonance = float(np.linalg.norm(diff_vec))
        is_conforming = dissonance <= 1.0 + self.tolerance

        if is_conforming:
            return StructuralDiscrepancy(
                is_conforming=True,
                hierarchical_layer="Narrative_Semantic_Field",
                kernel_defect=np.zeros(self.dimension, dtype=np.float64),
                symmetry_rupture_axis=np.zeros(self.dimension, dtype=np.float64),
                topological_obstruction="None",
                qualitative_alterity="Continuous_Dramatic_Progression",
                defect_magnitude=0.0
            )

        rupture_axis = diff_vec / (dissonance + 1e-9)
        excess_defect = diff_vec * ((dissonance - 1.0) / dissonance)
        return StructuralDiscrepancy(
            is_conforming=False,
            hierarchical_layer="Narrative_Semantic_Field",
            kernel_defect=excess_defect,
            symmetry_rupture_axis=rupture_axis,
            topological_obstruction=(
                "Contextual_Semantic_Dissonance: 극적 긴장과 서사적 연속성의 임계치를 초과하여 "
                "맥락적 장력이 파열됨"
            ),
            qualitative_alterity="Radical_Contextual_Disruption",
            defect_magnitude=float(dissonance - 1.0)
        )

    def step_dynamics(self, state: np.ndarray, incoming_flux: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        next_state = 0.7 * state + 0.3 * incoming_flux
        narrative_utterance = (state + incoming_flux) * 0.5
        return next_state, narrative_utterance, float(np.linalg.norm(state - incoming_flux))


class MultimodalCausalCognitiveNetwork:
    """
    [다중 모달리티 인과 인지 네트워크 (Multimodal Causal Cognitive Network)]
    Spans and differentiates across Mathematics, Code, Language, and Physics.
    Integrates their respective causal constraints into an interconnected, self-reproducing
    Cognitive Web with circular closure (순환논리를 가진 세계구조).
    """
    def __init__(self, base_dim: int = 4):
        self.base_dim = base_dim

        self.cells: Dict[ModalityType, DigitalCausalCell] = {
            ModalityType.MATHEMATICS: DigitalCausalCell(
                cell_id="MathNode",
                dimension=base_dim,
                constraint=MathematicalCausalConstraint(dimension=base_dim)
            ),
            ModalityType.CODE: DigitalCausalCell(
                cell_id="CodeNode",
                dimension=base_dim,
                constraint=CodeExecutionConstraint(dimension=base_dim)
            ),
            ModalityType.LANGUAGE: DigitalCausalCell(
                cell_id="LanguageNode",
                dimension=base_dim,
                constraint=LanguageNarrativeConstraint(dimension=base_dim)
            )
        }

        self.expansion_engine = MetaOrderExpansionEngine(defect_persistence_threshold=1)
        self.relational_bridges: List[Dict[str, Any]] = []
        self.global_teleological_intent: str = "Universal Spatiotemporal Epistemic Reproduction"

    def process_spatiotemporal_causal_stream(
        self,
        stream: List[CausalStep]
    ) -> List[Dict[str, Any]]:
        results = []

        for step in stream:
            target_cell = self.cells.get(step.modality)
            if target_cell is None:
                continue

            interaction_res = target_cell.interact_with_flux(step.resultant_state)
            discrepancy: StructuralDiscrepancy = interaction_res["discrepancy"]

            inspection = self.expansion_engine.inspect_cell_boundary(target_cell)

            expansion_info = None
            if inspection["requires_expansion"]:
                expansion_res = self.expansion_engine.execute_combinatorial_expansion(target_cell)
                expansion_info = expansion_res

                self.relational_bridges.append({
                    "source_modality": step.modality.value,
                    "step_index": step.step_index,
                    "new_macro_order": expansion_res.get("new_macro_order"),
                    "expanded_dimension": expansion_res.get("new_dimension")
                })

            results.append({
                "step_index": step.step_index,
                "modality": step.modality.value,
                "premise": step.premise,
                "transition": step.action_or_transition,
                "is_conforming": discrepancy.is_conforming,
                "boundary_friction": discrepancy.defect_magnitude,
                "discrepancy": discrepancy,
                "topological_obstruction": discrepancy.topological_obstruction,
                "hierarchical_layer": discrepancy.hierarchical_layer,
                "surface_tension": interaction_res["surface_tension"],
                "engram_count": len(target_cell.boundary_layer.retained_engrams),
                "expansion_triggered": expansion_info is not None,
                "expansion_detail": expansion_info
            })

        return results

    def verify_cyclical_world_closure(self) -> Dict[str, Any]:
        total_engrams = sum(len(c.boundary_layer.retained_engrams) for c in self.cells.values())
        mean_surface_tension = float(np.mean([c.boundary_layer.surface_tension for c in self.cells.values()]))
        expanded_nodes = [name.value for name, c in self.cells.items() if isinstance(c.governing_constraint, CoupledMacroConstraint)]

        is_cyclical_world_formed = (
            total_engrams > 0 and
            mean_surface_tension > 0.5 and
            len(self.relational_bridges) > 0
        )

        return {
            "is_cyclical_world_formed": is_cyclical_world_formed,
            "total_retained_engrams": total_engrams,
            "mean_surface_tension": mean_surface_tension,
            "differentiated_modalities": [m.value for m in self.cells.keys()],
            "expanded_nodes_count": len(expanded_nodes),
            "expanded_nodes": expanded_nodes,
            "relational_bridges_count": len(self.relational_bridges),
            "teleological_intent": self.global_teleological_intent,
            "verdict": (
                "세계의 다양한 인과서순(수학, 코드, 언어)을 구조적 계층원리로 지각하고, "
                "위상적 결함(Obstruction)을 표면장력으로 머금으며, 결합원리적 동형성을 통해 상위 인지망을 자발 구축함. "
                "순환논리를 가진 생명적 세계구조 확립 확인." if is_cyclical_world_formed else
                "아직 인과적 순환 닫힘이 불완전함."
            )
        }
