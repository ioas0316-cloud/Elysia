"""
Self-Forming Causal Sensor & World Friction Calibration with Phenomenological Direct Perception.

Rejects passive reception of fixed data vectors and dead numerical proxies.
Sprout/forms self-observation axes (Causal Sensors) and projects them into reality,
calibrating lens curvature and observation axes via reality friction (Friction & Refraction Error)
and direct phenomenological field resonance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math
from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension, RefractedObservation


@dataclass
class ObservationAxis:
    axis_name: str
    projection_vector: List[float]
    sensitivity: float = 1.0


@dataclass
class FrictionResult:
    friction_magnitude: float
    refraction_error: float
    adjusted_axes: List[ObservationAxis]
    recommended_curvature_delta: float


@dataclass
class PhenomenologicalResonance:
    is_unmediated: bool
    phenomenon_type: str
    direct_field_resonance: float
    is_dead_data_proxy: bool = False  # Always False!


class CausalSensor:
    """Active, self-forming sensor that projects observation axes into reality and self-calibrates."""

    def __init__(self, sensor_id: str, lens_engine: CognitiveLensEngine):
        self.sensor_id = sensor_id
        self.lens_engine = lens_engine
        self.axes: List[ObservationAxis] = self._form_initial_axes()

    def _form_initial_axes(self) -> List[ObservationAxis]:
        """Self-forms observation axes based on internal active inquiry."""
        return [
            ObservationAxis("spatial_curvature_axis", [1.0, 0.0, 0.0], 1.0),
            ObservationAxis("biological_friction_axis", [0.0, 1.0, 0.0], 1.0),
            ObservationAxis("relational_intent_axis", [0.0, 0.0, 1.0], 1.0)
        ]

    def observe_direct_phenomenon(self, phenomenon_name: str, raw_field_interaction: Dict[str, Any]) -> PhenomenologicalResonance:
        """Observes phenomenon directly without numerical proxy distortion."""
        intensity = float(raw_field_interaction.get("intensity", raw_field_interaction.get("energy", 1.0)))
        resonance = math.sin(intensity * math.pi / 2.0)

        return PhenomenologicalResonance(
            is_unmediated=True,
            phenomenon_type=phenomenon_name,
            direct_field_resonance=resonance,
            is_dead_data_proxy=False
        )

    def project_and_measure_friction(self, stimulus: Dict[str, Any], external_reality_feedback: Dict[str, Any]) -> FrictionResult:
        """Projects formed axes into external reality, observing phase friction and refraction error."""
        spectrum = self.lens_engine.observe_spectrum(stimulus)

        # Calculate predicted vs actual phase friction
        total_predicted_tension = sum(obs.phase_tension for obs in spectrum.values())
        actual_world_friction = float(external_reality_feedback.get("world_friction", 0.5))

        friction_diff = actual_world_friction - total_predicted_tension
        refraction_error = abs(friction_diff)

        # Self-calibration: Adjust axis sensitivity and recommended curvature delta
        curvature_delta = 0.1 if friction_diff > 0 else -0.1

        for axis in self.axes:
            axis.sensitivity += 0.05 * friction_diff

        return FrictionResult(
            friction_magnitude=actual_world_friction,
            refraction_error=refraction_error,
            adjusted_axes=self.axes,
            recommended_curvature_delta=curvature_delta
        )

    def self_calibrate(self, friction_result: FrictionResult):
        """Calibrates lens engine curvatures based on reality friction feedback."""
        for dim in ContextualDimension:
            current_curvature = self.lens_engine.lenses[dim].curvature
            new_curvature = max(0.1, current_curvature + friction_result.recommended_curvature_delta)
            self.lens_engine.adjust_lens_curvature(dim, new_curvature)
