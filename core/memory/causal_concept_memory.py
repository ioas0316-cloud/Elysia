"""
Causal Concept Memory
=====================

Concept understanding is stored as causal resonance, not as a label table.
An entity such as an apple becomes intelligible when repeated observations
settle into invariants, allowed variations, and reality-grounded feature
attractors inside domain-specific sense layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import hashlib
import numpy as np

from core.topology.causal_judgment_gear import (
    CausalGearState,
    CausalJudgmentGear,
    CausalJudgmentTrace,
    GearRegion,
)
from core.topology.causal_sense_layers import CausalSenseLayerEngine
from core.topology.informational_phase_observation import ChromaticVector


class VariantReality(Enum):
    REAL_GROUNDED = "real_grounded"
    IMAGINABLE = "imaginable"
    CONTRADICTED = "contradicted"


@dataclass
class FeatureAttractor:
    """A feature remembered as a grounded causal attractor."""

    feature_axis: str
    feature_value: str
    domain_layer: str
    support_mass: float = 0.0
    imagination_mass: float = 0.0
    contradiction_tension: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.feature_axis}:{self.feature_value}"


@dataclass
class CausalObservation:
    """One world or imagination encounter with a concept."""

    observation_id: str
    concept_id: str
    features: Dict[str, str]
    process_context: Dict[str, Any]
    evidence_strength: float = 1.0
    is_real_world: bool = True


@dataclass
class CausalConceptDefinition:
    """
    The internalized causal definition of a concept.

    Invariants state why the concept remains itself. Feature attractors state
    how it may vary without losing its identity.
    """

    concept_id: str
    label: str
    invariants: List[str] = field(default_factory=list)
    feature_attractors: Dict[str, FeatureAttractor] = field(default_factory=dict)
    causal_history: List[CausalJudgmentTrace] = field(default_factory=list)
    definition_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))


@dataclass
class VariantAssessment:
    """Assessment of whether a proposed variant is reality-grounded or imagined."""

    concept_id: str
    variant_name: str
    reality: VariantReality
    grounded_support: float
    imagination_support: float
    contradiction_tension: float
    judgment_trace: CausalJudgmentTrace
    evidence: Dict[str, float]


class CausalConceptMemory:
    """
    Learns concepts as cause-process-result resonance.

    The memory can later discern whether a variant is a known real form, a
    plausible imagined form, or a contradiction against the concept's own
    causal invariants.
    """

    def __init__(self, vector_dim: int = 8):
        self.vector_dim = vector_dim
        self.sense_layers = CausalSenseLayerEngine.with_foundational_layers(target_dimension=vector_dim)
        self.judgment_gear = CausalJudgmentGear()
        self.concepts: Dict[str, CausalConceptDefinition] = {}

    def learn_observation(
        self,
        observation: CausalObservation,
        label: Optional[str] = None,
        invariants: Optional[List[str]] = None,
    ) -> CausalConceptDefinition:
        concept = self.concepts.setdefault(
            observation.concept_id,
            CausalConceptDefinition(
                concept_id=observation.concept_id,
                label=label or observation.concept_id,
                invariants=list(invariants or []),
                definition_vector=np.zeros(self.vector_dim, dtype=np.float32),
            ),
        )

        for invariant in invariants or []:
            if invariant not in concept.invariants:
                concept.invariants.append(invariant)

        feature_vector = self._features_to_vector(observation.features)
        process_vector = _normalize(self._process_to_vector(observation.process_context) + feature_vector * 0.5)
        result_vector = _normalize(concept.definition_vector + feature_vector)

        cause = CausalGearState(
            GearRegion.CAUSE,
            observation.observation_id,
            feature_vector,
            resistance=0.2 if observation.is_real_world else 0.55,
            conductance=float(np.clip(observation.evidence_strength, 0.05, 1.0)),
            chromatic=ChromaticVector(flux=1.0, order=1.1, entropy=0.15 if observation.is_real_world else 0.55),
        )
        process = CausalGearState(
            GearRegion.PROCESS,
            "concept_internalization",
            process_vector,
            resistance=0.3,
            conductance=0.8,
            chromatic=ChromaticVector(flux=0.9, order=1.2, entropy=0.25),
        )
        result = CausalGearState(
            GearRegion.RESULT,
            concept.label,
            result_vector,
            resistance=0.25,
            conductance=0.85,
            chromatic=ChromaticVector(flux=1.0, order=1.0, entropy=0.2),
        )
        trace = self.judgment_gear.evaluate(cause, process, result)
        concept.causal_history.append(trace)
        concept.definition_vector = _normalize(concept.definition_vector + feature_vector * observation.evidence_strength)

        domain_layer = str(observation.process_context.get("domain_layer", "language_layer"))
        for axis, value in observation.features.items():
            key = f"{axis}:{value}"
            attractor = concept.feature_attractors.setdefault(
                key,
                FeatureAttractor(feature_axis=axis, feature_value=value, domain_layer=domain_layer),
            )
            if observation.is_real_world:
                grounding_mass = observation.evidence_strength * (0.25 + 0.75 * trace.structural_validity)
                attractor.support_mass += grounding_mass
            else:
                attractor.imagination_mass += observation.evidence_strength * trace.structural_validity
            attractor.contradiction_tension = max(0.0, attractor.contradiction_tension - 0.05)

        return concept

    def assess_variant(
        self,
        concept_id: str,
        variant_name: str,
        proposed_features: Dict[str, str],
        process_context: Optional[Dict[str, Any]] = None,
    ) -> VariantAssessment:
        if concept_id not in self.concepts:
            raise KeyError(f"Unknown concept: {concept_id}")

        concept = self.concepts[concept_id]
        context = process_context or {}

        grounded_support = 0.0
        imagination_support = 0.0
        contradiction_tension = 0.0

        observed_axes = {a.feature_axis for a in concept.feature_attractors.values()}
        for axis, value in proposed_features.items():
            key = f"{axis}:{value}"
            attractor = concept.feature_attractors.get(key)
            if attractor is not None:
                grounded_support += attractor.support_mass
                imagination_support += attractor.imagination_mass
                contradiction_tension += attractor.contradiction_tension
                continue

            axis_is_variable = axis in observed_axes or axis in set(context.get("variable_axes", []))
            if axis_is_variable:
                imagination_support += 0.35
                contradiction_tension += 0.15
            else:
                contradiction_tension += 0.65

        for violated in context.get("violates_invariants", []):
            if violated in concept.invariants:
                contradiction_tension += 1.0

        feature_vector = self._features_to_vector(proposed_features)
        cause = CausalGearState(GearRegion.CAUSE, variant_name, feature_vector, resistance=0.35, conductance=0.7)
        process = CausalGearState(
            GearRegion.PROCESS,
            "variant_discernment",
            _normalize(self._process_to_vector(context) + feature_vector * 0.5),
            resistance=0.4,
            conductance=0.75,
        )
        result = CausalGearState(
            GearRegion.RESULT,
            concept.label,
            concept.definition_vector,
            resistance=float(np.clip(0.25 + contradiction_tension * 0.2, 0.0, 1.0)),
            conductance=float(np.clip(0.85 + grounded_support * 0.05, 0.05, 1.0)),
        )
        judgment = self.judgment_gear.evaluate(cause, process, result)

        reality_score = grounded_support * (0.25 + 0.75 * judgment.structural_validity)
        imagination_score = imagination_support * (1.0 - min(1.0, contradiction_tension * 0.5))

        if contradiction_tension >= 1.0:
            reality = VariantReality.CONTRADICTED
        elif reality_score > 0.15:
            reality = VariantReality.REAL_GROUNDED
        else:
            reality = VariantReality.IMAGINABLE

        return VariantAssessment(
            concept_id=concept_id,
            variant_name=variant_name,
            reality=reality,
            grounded_support=float(grounded_support),
            imagination_support=float(imagination_support),
            contradiction_tension=float(contradiction_tension),
            judgment_trace=judgment,
            evidence={
                "reality_score": float(reality_score),
                "imagination_score": float(imagination_score),
                "structural_validity": float(judgment.structural_validity),
            },
        )

    def _features_to_vector(self, features: Dict[str, str]) -> np.ndarray:
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        for axis, value in sorted(features.items()):
            digest = hashlib.sha256(f"{axis}:{value}".encode("utf-8")).digest()
            for i, byte in enumerate(digest[: self.vector_dim]):
                vector[i] += (byte / 255.0) * 2.0 - 1.0
        return _normalize(vector)

    def _process_to_vector(self, process_context: Dict[str, Any]) -> np.ndarray:
        if not process_context:
            return _normalize(np.ones(self.vector_dim, dtype=np.float32))
        return self._features_to_vector({str(k): str(v) for k, v in process_context.items()})


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm <= 1e-8:
        return arr
    return arr / norm
