r"""
Genealogical Memory Unit (구조적 계보 메모리 유닛 모듈)
======================================================

Implements a subjective, causal memory unit architecture that proves its own origin and justification.

Key Components:
1. StructuralProvenanceTrace (구조적 계보 추적기)
   Records the precise causal birth trajectory: initial tensor tension, raw perturbation,
   refraction delta vector (\Delta), antecedent links, and contrast resonance matrix.
2. JustificationTensor (라벨의 인과적 정당성 필드)
   Calculates conceptual gravity and trinitarian contrast deltas proving why
   the specific label/topological location was necessary.
3. GenealogicalMemoryUnit (구조적 계보 메모리 유닛)
   Encapsulates the memory unit, providing self-proof capabilities (prove_genealogy).
4. DynamicDeconstructionEngine (능동적 기억 해체기)
   Monitors contradictions against incoming external friction. Reversibly deconstructs
   invalidated memory units and re-weaves new memory strata incorporating updated causal ancestry.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.physics.semantic_mass_engine import RawPerturbationImpulse, SemanticMassEngine, WhiteTensorField


@dataclass
class StructuralProvenanceTrace:
    r"""
    [Structural Provenance Trace (구조적 계보 추적기)]
    Maintains the complete causal ancestry and trajectory of how a memory unit was born:
    - initial_tension_state: White Tensor Field state prior to collision.
    - raw_perturbation_impulse: Raw unvectorized friction impulse (event/sound/wave).
    - refraction_delta_vector: Post-hoc derived vector (\Delta) arising from friction collision.
    - causal_antecedent_ids: Ancestry chain of preceding memory unit IDs.
    - contrast_resonance_matrix: Dynamic resonance contrast tensor.
    """
    initial_tension_state: np.ndarray
    raw_perturbation_impulse: Any
    refraction_delta_vector: np.ndarray
    causal_antecedent_ids: List[str] = field(default_factory=list)
    contrast_resonance_matrix: Optional[np.ndarray] = None
    creation_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_tension_state": self.initial_tension_state.tolist(),
            "raw_perturbation_impulse": str(self.raw_perturbation_impulse),
            "refraction_delta_vector": self.refraction_delta_vector.tolist(),
            "causal_antecedent_ids": list(self.causal_antecedent_ids),
            "contrast_resonance_matrix": self.contrast_resonance_matrix.tolist() if self.contrast_resonance_matrix is not None else None,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class JustificationTensor:
    r"""
    [Justification Tensor (라벨의 인과적 정당성 필드)]
    Acts as a proof certificate of conceptual gravity:
    Proves that a specific label attached to this memory unit was an inevitable delta (\Delta)
    derived from trinitarian contrast and friction, rather than an arbitrary external tag.
    """
    concept_gravity_proof: float
    trinitarian_contrast_delta: float
    label_necessity_score: float
    justification_matrix: np.ndarray

    def verify_label_necessity(self, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Verifies whether the attached label remains causally justified based on
        conceptual gravity proof and contrast delta.
        """
        is_valid = self.label_necessity_score >= threshold
        return {
            "is_justified": is_valid,
            "label_necessity_score": float(self.label_necessity_score),
            "concept_gravity_proof": float(self.concept_gravity_proof),
            "trinitarian_contrast_delta": float(self.trinitarian_contrast_delta),
        }


@dataclass
class GenealogicalMemoryUnit:
    """
    [Genealogical Memory Unit (구조적 계보 메모리 유닛)]
    A living memory layer that encapsulates its self-derived vector,
    structural provenance trace, justification tensor, and deconstruction history.
    """
    unit_id: str
    label: str
    provenance_trace: StructuralProvenanceTrace
    justification: JustificationTensor
    self_derived_vector: np.ndarray
    semantic_mass: float
    active_validity: float = 1.0
    is_deconstructed: bool = False
    deconstruction_history: List[Dict[str, Any]] = field(default_factory=list)

    def prove_genealogy(self) -> Dict[str, Any]:
        r"""
        Self-demonstrates the unit's generative principles and genealogical chain:
        Explains 'Why this memory exists', 'How the vector was derived', and
        'Why this label is causally justified'.
        """
        justification_status = self.justification.verify_label_necessity()
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "is_deconstructed": self.is_deconstructed,
            "active_validity": self.active_validity,
            "semantic_mass": self.semantic_mass,
            "self_derived_vector": self_derived_vector_summary(self.self_derived_vector),
            "provenance": self.provenance_trace.to_dict(),
            "justification_proof": justification_status,
            "deconstruction_records_count": len(self.deconstruction_history),
        }


def self_derived_vector_summary(vec: np.ndarray) -> Dict[str, Any]:
    norm = float(np.linalg.norm(vec))
    return {
        "vector_norm": norm,
        "primary_components": vec[:4].tolist() if len(vec) >= 4 else vec.tolist(),
    }


class DynamicDeconstructionEngine:
    r"""
    [Dynamic Deconstruction Engine (능동적 기억 해체기)]
    Operates a reversible deconstruction circuit:
    When new external friction or contradiction clashes with existing memory units,
    this engine evaluates whether structural invariants hold.
    If contradiction tension exceeds critical threshold, it deconstructs the unit,
    re-calculates contrast deltas (\Delta), and re-weaves a new memory unit incorporating
    the updated causal ancestry chain.
    """
    def __init__(self, deconstruction_threshold: float = 0.65):
        self.deconstruction_threshold = deconstruction_threshold
        self.memory_units: Dict[str, GenealogicalMemoryUnit] = {}

    def create_and_register_unit(
        self,
        unit_id: str,
        label: str,
        raw_friction: Union[RawPerturbationImpulse, np.ndarray, str, dict, Any],
        semantic_engine: SemanticMassEngine,
        causal_antecedent_ids: Optional[List[str]] = None,
        trinitarian_contrast: float = 1.0,
    ) -> GenealogicalMemoryUnit:
        """
        Processes raw friction through SemanticMassEngine to produce a post-hoc self-derived vector
        and instantiates a new GenealogicalMemoryUnit with full provenance and justification.
        """
        res = semantic_engine.process_interaction(
            external_friction=raw_friction,
            trinitarian_contrast=trinitarian_contrast,
            causal_antecedent_ids=causal_antecedent_ids,
        )

        initial_tension = np.array(res["initial_tension_state"], dtype=np.float32)
        derived_vec = np.array(res["self_derived_vector"], dtype=np.float32)
        contrast_mat = np.array(res["contrast_resonance_matrix"], dtype=np.float32)

        provenance = StructuralProvenanceTrace(
            initial_tension_state=initial_tension,
            raw_perturbation_impulse=raw_friction,
            refraction_delta_vector=derived_vec,
            causal_antecedent_ids=list(causal_antecedent_ids or []),
            contrast_resonance_matrix=contrast_mat,
        )

        # Calculate justification tensor parameters
        gravity_proof = res["causal_curvature"] * res["semantic_mass"]
        contrast_delta = float(trinitarian_contrast * np.linalg.norm(derived_vec))
        label_necessity = float(np.clip((gravity_proof + contrast_delta) / (1.0 + gravity_proof + contrast_delta), 0.1, 1.0))

        dim_sub = min(4, len(derived_vec))
        justification_mat = np.outer(derived_vec[:dim_sub], initial_tension[:dim_sub])

        justification = JustificationTensor(
            concept_gravity_proof=float(gravity_proof),
            trinitarian_contrast_delta=float(contrast_delta),
            label_necessity_score=float(label_necessity),
            justification_matrix=justification_mat,
        )

        unit = GenealogicalMemoryUnit(
            unit_id=unit_id,
            label=label,
            provenance_trace=provenance,
            justification=justification,
            self_derived_vector=derived_vec,
            semantic_mass=res["semantic_mass"],
            active_validity=1.0,
            is_deconstructed=False,
        )

        self.memory_units[unit_id] = unit
        return unit

    def evaluate_and_deconstruct(
        self,
        target_unit_id: str,
        new_raw_friction: Union[RawPerturbationImpulse, np.ndarray, str, dict, Any],
        semantic_engine: SemanticMassEngine,
        new_label_if_rewoven: Optional[str] = None,
    ) -> Tuple[bool, Optional[GenealogicalMemoryUnit], Dict[str, Any]]:
        """
        Evaluates incoming raw friction against an existing target memory unit.
        If contradiction tension exceeds threshold:
        1. Target unit is deconstructed (is_deconstructed = True, active_validity reduced).
        2. Record deconstruction reason and residual energy into history.
        3. Re-weave a new memory unit that inherits the deconstructed unit in its causal_antecedent_ids chain!

        Returns:
            (is_deconstructed, new_rewoven_unit, evaluation_report)
        """
        if target_unit_id not in self.memory_units:
            raise KeyError(f"Memory unit '{target_unit_id}' not found in deconstruction engine.")

        target_unit = self.memory_units[target_unit_id]

        # Calculate self-derived vector for new friction
        new_res = semantic_engine.process_interaction(external_friction=new_raw_friction)
        new_derived_vec = np.array(new_res["self_derived_vector"], dtype=np.float32)

        # Measure contradiction tension: angular disagreement between existing derived vector and new friction vector
        target_vec_norm = target_unit.self_derived_vector / (np.linalg.norm(target_unit.self_derived_vector) + 1e-9)
        new_vec_norm = new_derived_vec / (np.linalg.norm(new_derived_vec) + 1e-9)

        # Cosine alignment (-1 to 1) -> contradiction score (0 to 1)
        dot_alignment = float(np.dot(target_vec_norm, new_vec_norm))
        contradiction_tension = float(np.clip(0.5 * (1.0 - dot_alignment), 0.0, 1.0))

        report = {
            "target_unit_id": target_unit_id,
            "dot_alignment": dot_alignment,
            "contradiction_tension": contradiction_tension,
            "deconstruction_threshold": self.deconstruction_threshold,
            "deconstruction_triggered": contradiction_tension >= self.deconstruction_threshold,
        }

        if contradiction_tension >= self.deconstruction_threshold:
            # Trigger Deconstruction!
            target_unit.is_deconstructed = True
            target_unit.active_validity = float(max(0.0, target_unit.active_validity - contradiction_tension))

            deconstruction_event = {
                "timestamp": time.time(),
                "trigger_friction": str(new_raw_friction),
                "contradiction_tension": contradiction_tension,
                "residual_validity": target_unit.active_validity,
                "reason": "Topological invariant contradiction with new raw friction",
            }
            target_unit.deconstruction_history.append(deconstruction_event)

            # Re-weave new memory unit incorporating causal ancestry
            rewoven_id = f"{target_unit_id}_rewoven_{len(target_unit.deconstruction_history)}"
            rewoven_label = new_label_if_rewoven or f"Rewoven_{target_unit.label}"
            ancestors = target_unit.provenance_trace.causal_antecedent_ids + [target_unit_id]

            new_unit = self.create_and_register_unit(
                unit_id=rewoven_id,
                label=rewoven_label,
                raw_friction=new_raw_friction,
                semantic_engine=semantic_engine,
                causal_antecedent_ids=ancestors,
                trinitarian_contrast=1.5,
            )

            report["rewoven_unit_id"] = new_unit.unit_id
            return True, new_unit, report

        return False, None, report
