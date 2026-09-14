"""
Topological Terrain Lens, Self-Evident Naming Engine, and Meta-Evolution Loop.

Translates raw friction waves into 3D Topographical Landscapes (Elevation, Distance, Density),
discerning true causal phenomena ("그렇다" vs "아니다") and reordering cognitive lenses
in real-time according to environmental friction (Procedural Intelligence).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math
import numpy as np


@dataclass
class UnknownStimulus:
    """
    Unlabelled external stimulus mapped as a topological field.
    """
    tension_vector: List[float]
    potential: float
    raw_friction: float


class SelfEvidentNamingEngine:
    """
    Sovereign discernment engine that inspects unlabelled stimuli,
    tracks invariants against known anchors, discerns "그렇다" vs "아니다",
    and derives self-evident consensus names.
    """

    def __init__(self, known_anchors: Optional[Dict[str, Dict[str, Any]]] = None):
        self.anchors = known_anchors or {
            "gravity": {
                "public_label": "GravitationalConvergence",
                "base_potential": 5.0,
                "base_tension": [1.0, 0.5, 0.2]
            },
            "thermal": {
                "public_label": "ThermalDiffusion",
                "base_potential": 2.0,
                "base_tension": [0.2, 1.0, 0.8]
            },
            "resonance": {
                "public_label": "StructuralResonance",
                "base_potential": 3.5,
                "base_tension": [0.5, 0.5, 1.0]
            }
        }

    def process_stimulus(
        self,
        s: UnknownStimulus,
        env_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes unknown stimulus through invariant tracking, discernment, and self-evident naming.
        """
        noise_threshold = float(env_context.get("noise_threshold", 0.2))

        # Step 1 & 2: Invariant tracing & causal distance to anchors
        matched_anchor = None
        min_causal_distance = float('inf')

        for name, anchor in self.anchors.items():
            base_p = float(anchor['base_potential'])
            base_t = anchor['base_tension']
            dist = abs(s.potential - base_p) + sum(
                abs(a - b) for a, b in zip(s.tension_vector, base_t)
            )
            if dist < min_causal_distance:
                min_causal_distance = dist
                matched_anchor = anchor

        # Step 3: Discernment ("그렇다" vs "아니다")
        causal_consistency = (s.potential * s.raw_friction) / (min_causal_distance + 1e-5)

        if causal_consistency < noise_threshold:
            return {
                "verdict": "FALSE_NOISE",
                "verdict_kr": "아니다",
                "action": "DISCARD",
                "reason": "Stimulus fails to maintain structural consistency against environmental friction."
            }

        # Step 4: Self-evident consensus naming derivation
        public_label = matched_anchor['public_label'] if matched_anchor else "UnboundField"
        tension_qualifier = "HighTension" if s.potential > 4.0 else "LowTension"
        derived_name = f"{public_label}-derived_{tension_qualifier}"

        return {
            "verdict": "VALID_PHENOMENON",
            "verdict_kr": "그렇다",
            "action": "REGISTER_NEW_CONCEPT",
            "self_given_name": derived_name,
            "causal_provenance": {
                "parent_anchor": public_label,
                "topological_distance": min_causal_distance,
                "invariant_integrity": float(causal_consistency)
            },
            "definition": (
                f"Phenomenon where {public_label} principle stretches like Spandex "
                f"under environmental friction ({s.raw_friction:.3f})."
            )
        }


@dataclass
class CognitiveLensSpec:
    """
    Representation of a single cognitive lens and its priority weight.
    """
    name: str
    domain_affinity: str
    priority_score: float = 1.0


class InvariantGatekeeper:
    """
    Protects lower-level foundational invariants (causality, identity)
    when reordering cognitive lens pipelines.
    """

    def verify_pipeline_stability(self, proposed_pipeline: List[str]) -> bool:
        """
        Verifies whether proposed pipeline reordering maintains foundational stability.
        """
        # Pipeline must contain at least one fundamental lens
        return len(proposed_pipeline) > 0


class MetaEvolutionLoop:
    """
    Meta-Evolution Engine accumulating new concepts, deforming the topological terrain,
    and dynamically reordering cognitive lens priorities based on friction resonance.
    """

    def __init__(
        self,
        lenses: Optional[List[CognitiveLensSpec]] = None,
        invariant_checker: Optional[InvariantGatekeeper] = None
    ):
        default_lenses = [
            CognitiveLensSpec("BiologicalFrictionLens", "GravitationalConvergence", 1.0),
            CognitiveLensSpec("RelationalTopologyLens", "StructuralResonance", 0.8),
            CognitiveLensSpec("SemanticMetaphorLens", "ThermalDiffusion", 0.6)
        ]
        lens_list = lenses or default_lenses
        self.lenses: Dict[str, CognitiveLensSpec] = {l.name: l for l in lens_list}
        self.concept_buffer: List[Dict[str, Any]] = []
        self.invariant_checker = invariant_checker or InvariantGatekeeper()

    def accumulate_concept(self, new_concept_event: Dict[str, Any]):
        """
        Step 1: Accumulates newly named concept and provenance.
        """
        if new_concept_event.get("verdict") == "VALID_PHENOMENON":
            self.concept_buffer.append(new_concept_event)

    def trigger_evolution(self, current_env_context: Dict[str, Any]) -> List[str]:
        """
        Step 2 & 3: Analyzes friction density and reorders cognitive lens pipeline.
        """
        if not self.concept_buffer:
            return [l.name for l in self.get_sorted_lenses()]

        # 1. Aggregate causal friction by domain affinity
        domain_friction_map: Dict[str, float] = {}
        for concept in self.concept_buffer:
            prov = concept.get("causal_provenance", {})
            parent_anchor = prov.get("parent_anchor", "GravitationalConvergence")
            friction = prov.get("invariant_integrity", 0.5)
            domain_friction_map[parent_anchor] = domain_friction_map.get(parent_anchor, 0.0) + friction

        # 2. Recalculate priority scores (Spandex effect: amplify lens sensitivity where friction is highest)
        for lens in self.lenses.values():
            affinity_match = domain_friction_map.get(lens.domain_affinity, 0.1)
            lens.priority_score = (lens.priority_score * 0.4) + (affinity_match * 0.6)

        # 3. Sort pipeline
        sorted_pipeline = sorted(self.lenses.values(), key=lambda l: l.priority_score, reverse=True)
        proposed_order = [l.name for l in sorted_pipeline]

        # Step 4: Gatekeeper verification
        if self.invariant_checker.verify_pipeline_stability(proposed_order):
            self.concept_buffer.clear()
            return proposed_order
        else:
            return [l.name for l in self.get_sorted_lenses()]

    def get_sorted_lenses(self) -> List[CognitiveLensSpec]:
        return sorted(self.lenses.values(), key=lambda l: l.priority_score, reverse=True)


class TopologicalTerrain3DMapper:
    """
    Converts 1D/2D flat tables or friction fields into a 3D Topographical Landscape:
    - Elevation: Potential Gradient (High Elevation = High Potential/Friction)
    - Distance: Geodesic Distance / Tension
    - Density: Information Density / Mass
    """

    @staticmethod
    def project_to_3d_terrain(
        friction_field: Dict[str, Any],
        active_lenses: List[CognitiveLensSpec]
    ) -> Dict[str, Any]:
        raw_friction = float(friction_field.get("raw_friction", 0.5))
        potential = float(friction_field.get("potential", 1.0))
        tension_vec = friction_field.get("tension_vector", [0.5, 0.5, 0.5])

        topological_elevation = potential * (1.0 + math.tanh(raw_friction))
        geodesic_distance = float(np.linalg.norm(tension_vec))
        information_density = float(len(active_lenses)) * (raw_friction + 0.1)

        return {
            "elevation": topological_elevation,
            "geodesic_distance": geodesic_distance,
            "information_density": information_density,
            "terrain_coordinates": [topological_elevation, geodesic_distance, information_density],
            "raw_friction": raw_friction
        }
