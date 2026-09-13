"""
Causal Adaptation Lens & Environmental Discernment Engine.

Implements environmental adaptation and causal discernment:
- Processes physical reality friction (fluid/field potential differences, boundary tension, phase divergence).
- Distinguishes causal invariants (structural necessity) from causal variants (environmental noise/fluctuations).
- Self-calibrates internal potential fields and variable rotor phase angles based on friction gradients.
- Generates self-evident causal labels grounded in structural invariance rather than static dictionaries.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np

from core.lens.cognitive_lens_engine import (
    CognitiveLens,
    ContextualDimension,
    RefractedObservation,
    CognitiveLensEngine
)


@dataclass
class RealityFrictionWave:
    """Represents raw physical reality wave/field interactions."""
    fluid_potential_diff: float
    boundary_tension: float
    shear_stress: float
    collision_momentum: float
    frequency_spectrum: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])


@dataclass
class DiscernmentResult:
    """Outcome of causal discernment on environmental friction."""
    friction_magnitude: float
    phase_divergence: float
    causal_invariants: List[str]
    causal_variants: Dict[str, float]
    self_evident_label: str
    phase_angle_delta: float
    calibrated_potential: float


class CausalAdaptationLens(CognitiveLens):
    """
    Cognitive lens designed to process physical reality friction and adapt internal state
    through causal discernment.
    """

    def __init__(
        self,
        initial_tension: float = 1.0,
        min_tension: float = 0.1,
        max_tension: float = 10.0,
        rotor_phase_angle: float = 0.0
    ):
        super().__init__(ContextualDimension.BIOLOGICAL_FRICTION, curvature=1.0)
        self.initial_tension = initial_tension
        self.min_tension = min_tension
        self.max_tension = max_tension
        self.current_tension = initial_tension
        self.rotor_phase_angle = rotor_phase_angle  # Rotor angle theta in radians
        self.internal_potential_field = initial_tension * 1.5

    def refract_reality_wave(self, wave: RealityFrictionWave) -> DiscernmentResult:
        """
        Processes an incoming reality friction wave, discerning invariant causality from variants,
        and computing self-calibration phase shifts.
        """
        # Calculate raw friction magnitude and phase divergence
        raw_friction = (
            abs(wave.fluid_potential_diff) * 0.3 +
            abs(wave.boundary_tension) * 0.4 +
            abs(wave.shear_stress) * 0.2 +
            abs(wave.collision_momentum) * 0.1
        )

        # Phase divergence between external wave & internal rotor phase
        wave_phase = math.atan2(wave.boundary_tension, wave.fluid_potential_diff + 1e-6)
        phase_divergence = abs((wave_phase - self.rotor_phase_angle) % (2 * math.pi))

        # Dynamic tension adjustment within dynamic range [min_tension, max_tension]
        tension_gradient = math.tanh(raw_friction / max(self.current_tension, 1e-5))
        new_tension = self.current_tension * (1.0 + 0.2 * tension_gradient)
        self.current_tension = float(np.clip(new_tension, self.min_tension, self.max_tension))

        # Discerning Invariants vs Variants
        invariants = []
        variants = {}

        if abs(wave.boundary_tension) > 0.5:
            invariants.append("boundary_continuity")
        else:
            variants["boundary_fluctuation"] = float(wave.boundary_tension)

        if abs(wave.fluid_potential_diff) > 0.3:
            invariants.append("field_potential_gradient")
        else:
            variants["potential_noise"] = float(wave.fluid_potential_diff)

        if wave.shear_stress * wave.collision_momentum > 0.2:
            invariants.append("shear_momentum_coupling")
        else:
            variants["shear_dissipation"] = float(wave.shear_stress)

        # Self-evident labeling based on invariant topology
        if "boundary_continuity" in invariants and "field_potential_gradient" in invariants:
            label = "CausalBoundaryObject::SelfSustainingField"
        elif "boundary_continuity" in invariants:
            label = "CausalBoundaryObject::ElasticSurface"
        elif "field_potential_gradient" in invariants:
            label = "CausalFieldFlux::GradientFlow"
        else:
            label = "UnformedFrictionNoise"

        # Variable Rotor Phase Angle Rotation (Delta Theta)
        phase_angle_delta = 0.1 * math.sin(phase_divergence) * tension_gradient
        self.rotor_phase_angle = (self.rotor_phase_angle + phase_angle_delta) % (2 * math.pi)

        # Internal potential field self-calibration
        calibrated_potential = self.current_tension * math.cos(self.rotor_phase_angle)
        self.internal_potential_field = calibrated_potential

        return DiscernmentResult(
            friction_magnitude=raw_friction,
            phase_divergence=phase_divergence,
            causal_invariants=invariants,
            causal_variants=variants,
            self_evident_label=label,
            phase_angle_delta=phase_angle_delta,
            calibrated_potential=calibrated_potential
        )

    def refract(self, stimulus: Dict[str, Any]) -> RefractedObservation:
        """Standard CognitiveLens interface compliance."""
        wave = RealityFrictionWave(
            fluid_potential_diff=float(stimulus.get("fluid_potential_diff", stimulus.get("intensity", 0.5))),
            boundary_tension=float(stimulus.get("boundary_tension", stimulus.get("tension", 0.5))),
            shear_stress=float(stimulus.get("shear_stress", 0.2)),
            collision_momentum=float(stimulus.get("collision_momentum", 0.1))
        )
        discernment = self.refract_reality_wave(wave)

        refraction_angle = math.tanh(discernment.friction_magnitude) * (math.pi / 2.0)
        phase_tension = discernment.phase_divergence

        bound_weaving = {
            "friction_magnitude": discernment.friction_magnitude,
            "self_evident_label": discernment.self_evident_label,
            "current_tension": self.current_tension,
            "rotor_phase_angle": self.rotor_phase_angle,
            "causal_variants": discernment.causal_variants
        }

        return RefractedObservation(
            lens_type=self.dimension,
            refraction_angle=refraction_angle,
            phase_tension=phase_tension,
            bound_weaving=bound_weaving,
            causal_invariants=discernment.causal_invariants
        )


class EnvironmentalDiscernmentEngine:
    """
    Orchestrates environmental perception, adaptation, and discernment by coupling
    the CausalAdaptationLens with CognitiveLensEngine and external reality streams.
    """

    def __init__(
        self,
        lens_engine: Optional[CognitiveLensEngine] = None,
        initial_tension: float = 1.0
    ):
        self.lens_engine = lens_engine or CognitiveLensEngine()
        self.adaptation_lens = CausalAdaptationLens(initial_tension=initial_tension)
        # Register CausalAdaptationLens into lens engine under BIOLOGICAL_FRICTION
        self.lens_engine.lenses[ContextualDimension.BIOLOGICAL_FRICTION] = self.adaptation_lens

    def process_environmental_friction(self, wave: RealityFrictionWave) -> Tuple[DiscernmentResult, Dict[ContextualDimension, RefractedObservation]]:
        """
        Processes reality friction wave through CausalAdaptationLens and multi-dimensional CognitiveLensEngine.
        """
        discernment = self.adaptation_lens.refract_reality_wave(wave)

        stimulus = {
            "fluid_potential_diff": wave.fluid_potential_diff,
            "boundary_tension": wave.boundary_tension,
            "shear_stress": wave.shear_stress,
            "collision_momentum": wave.collision_momentum,
            "intensity": discernment.friction_magnitude
        }

        spectrum = self.lens_engine.observe_spectrum(stimulus)
        return discernment, spectrum

    def self_align_to_friction(self, discernment: DiscernmentResult):
        """
        Performs self-alignment of internal cognitive curvatures based on reality friction.
        """
        curvature_delta = 0.05 * math.tanh(discernment.friction_magnitude)
        for dim in ContextualDimension:
            current_curvature = self.lens_engine.lenses[dim].curvature
            new_curvature = float(np.clip(current_curvature + curvature_delta, 0.1, 5.0))
            self.lens_engine.adjust_lens_curvature(dim, new_curvature)
