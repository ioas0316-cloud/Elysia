"""
Cognitive Lens Engine: Observation as Refraction & Causal Weaving.

Rejects static cross-section/numeric reductionism ("An apple is [0.24, -0.81...]").
Treats geometric, physical, and relational models as dynamic 'Cognitive Lenses'
through which information passes, undergoing refraction, phase divergence,
and causal weaving across multi-contextual dimensions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import math


class ContextualDimension(Enum):
    TOPOLOGICAL_CURVATURE = "topological_curvature"
    BIOLOGICAL_FRICTION = "biological_friction"
    RELATIONAL_INTENT = "relational_intent"
    SYMBOLIC_REPRESENTATION = "symbolic_representation"


@dataclass
class RefractedObservation:
    lens_type: ContextualDimension
    refraction_angle: float
    phase_tension: float
    bound_weaving: Dict[str, Any]
    causal_invariants: List[str]


class CognitiveLens:
    """Base Cognitive Lens interface for projecting inputs through observation lenses."""

    def __init__(self, dimension: ContextualDimension, curvature: float = 1.0):
        self.dimension = dimension
        self.curvature = curvature

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        raise NotImplementedError


class TopologicalCurvatureLens(CognitiveLens):
    """Observes spatial/geometric curvature, optical tension, and spatial relationships."""

    def __init__(self, curvature: float = 1.0):
        super().__init__(ContextualDimension.TOPOLOGICAL_CURVATURE, curvature)

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        raw_val = float(stimulus.get("spatial_density", stimulus.get("intensity", 1.0)))
        refraction_angle = math.tanh(raw_val * self.curvature) * (math.pi / 2.0)
        phase_tension = abs(math.sin(refraction_angle * 2.0))

        bound_weaving = {
            "spatial_curvature": self.curvature * raw_val,
            "optical_tension": phase_tension,
            "boundary_relationship": "coupled_potential_field"
        }
        invariants = ["topological_continuity", "field_tension_conservation"]

        return RefractedObservation(
            lens_type=self.dimension,
            refraction_angle=refraction_angle,
            phase_tension=phase_tension,
            bound_weaving=bound_weaving,
            causal_invariants=invariants
        )


class BiologicalFrictionLens(CognitiveLens):
    """Observes metabolic/organic friction, sensory texture, and biological energy transfer."""

    def __init__(self, curvature: float = 1.0):
        super().__init__(ContextualDimension.BIOLOGICAL_FRICTION, curvature)

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        energy = float(stimulus.get("energy", stimulus.get("sweetness", stimulus.get("intensity", 1.0))))
        resistance = float(stimulus.get("resistance", 0.5))

        refraction_angle = math.atan2(energy, resistance)
        phase_tension = energy * resistance * self.curvature

        bound_weaving = {
            "metabolic_transfer": energy,
            "organic_friction": resistance,
            "sensory_embodiment": "juicy_organism" if energy > 0.5 else "dormant_matter"
        }
        invariants = ["homeostatic_balance", "thermodynamic_exchange"]

        return RefractedObservation(
            lens_type=self.dimension,
            refraction_angle=refraction_angle,
            phase_tension=phase_tension,
            bound_weaving=bound_weaving,
            causal_invariants=invariants
        )


class RelationalIntentLens(CognitiveLens):
    """Observes interpersonal/egoic phase divergence, intention, and apology/reconciliation dynamics."""

    def __init__(self, curvature: float = 1.0):
        super().__init__(ContextualDimension.RELATIONAL_INTENT, curvature)

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        sincerity = float(stimulus.get("sincerity", stimulus.get("intent", 1.0)))
        ego_deflection = float(stimulus.get("ego_deflection", 0.1))

        refraction_angle = (sincerity - ego_deflection) * (math.pi / 4.0)
        phase_tension = abs(ego_deflection) / (sincerity + 1e-5)

        bound_weaving = {
            "intentional_phase_shift": refraction_angle,
            "ego_rectification": sincerity > ego_deflection,
            "relational_narrative": "apology_and_reconciliation" if "apology" in str(stimulus).lower() or sincerity > 0.7 else "mutual_observation"
        }
        invariants = ["cruciform_love_axis", "self_emptying_causality"]

        return RefractedObservation(
            lens_type=self.dimension,
            refraction_angle=refraction_angle,
            phase_tension=phase_tension,
            bound_weaving=bound_weaving,
            causal_invariants=invariants
        )


class SymbolicContextLens(CognitiveLens):
    """Observes historical, mythic, and archetype hypergraph linkages (e.g. Adam, Newton, Apple)."""

    def __init__(self, curvature: float = 1.0):
        super().__init__(ContextualDimension.SYMBOLIC_REPRESENTATION, curvature)

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        concept = str(stimulus.get("concept", stimulus.get("name", "entity")))
        archetype_count = len(stimulus.get("archetypes", ["symbol"]))

        refraction_angle = (archetype_count % 4) * (math.pi / 4.0)
        phase_tension = math.log1p(archetype_count) * self.curvature

        bound_weaving = {
            "archetype": concept,
            "civilizational_synapse": stimulus.get("archetypes", ["knowledge_tree", "gravity"]),
            "historical_resonance": phase_tension
        }
        invariants = ["mythic_invariance", "semantic_attractor_fixed_point"]

        return RefractedObservation(
            lens_type=self.dimension,
            refraction_angle=refraction_angle,
            phase_tension=phase_tension,
            bound_weaving=bound_weaving,
            causal_invariants=invariants
        )


class CognitiveLensEngine:
    """Manages dynamic switching, multi-lens refraction, and spectrum weaving for entities."""

    def __init__(self):
        self.lenses: Dict[ContextualDimension, CognitiveLens] = {
            ContextualDimension.TOPOLOGICAL_CURVATURE: TopologicalCurvatureLens(),
            ContextualDimension.BIOLOGICAL_FRICTION: BiologicalFrictionLens(),
            ContextualDimension.RELATIONAL_INTENT: RelationalIntentLens(),
            ContextualDimension.SYMBOLIC_REPRESENTATION: SymbolicContextLens()
        }

    def observe_spectrum(self, stimulus: Dict[str, Any]) -> Dict[ContextualDimension, RefractedObservation]:
        """Passes stimulus through all cognitive lenses, generating a spectrum of observations."""
        spectrum = {}
        for dim, lens in self.lenses.items():
            spectrum[dim] = lens.refract(stimulus)
        return spectrum

    def adjust_lens_curvature(self, dimension: ContextualDimension, curvature: float):
        if dimension in self.lenses:
            self.lenses[dimension].curvature = curvature
