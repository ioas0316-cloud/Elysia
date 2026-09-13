"""
Causal Adaptation Lens & Environmental Discernment Engine.

Implements environmental adaptation and causal discernment through:
- Minimal Causal Operator (Dynamic Operator for Tension Convergence/Divergence).
- Topological Invariant Tensor Extraction (Field potential gradient, boundary winding, shear coupling).
- Causal Provenance Trajectory (Tracking Dialectical Thesis-Antithesis-Synthesis Lineage).
- Physical reality friction processing and self-evident validation.
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
class TopologicalInvariant:
    """
    Topological invariant metric capturing generative field conditions
    rather than static dictionary definitions.
    """
    name: str
    magnitude: float
    topological_dimension: int
    is_stable: bool
    curvature_index: float

    def validity_score(self) -> float:
        """Computes self-evident validity score of this invariant."""
        return float(np.clip(abs(self.magnitude) * (1.0 if self.is_stable else 0.5), 0.0, 1.0))


@dataclass
class ProvenanceStep:
    """
    A single dialectical transformation step in the causal lineage.
    """
    step_id: int
    thesis_wave_magnitude: float
    antithesis_internal_tension: float
    synthesis_calibrated_potential: float
    rotor_phase_angle: float
    phase_divergence: float
    invariants_extracted: List[str]
    validity_confidence: float


class CausalProvenanceTrajectory:
    """
    Records and validates the causal lineage (인과적 족보) of topological transformations
    over time through physical reality friction.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: List[ProvenanceStep] = []

    def record_step(
        self,
        wave_magnitude: float,
        internal_tension: float,
        calibrated_potential: float,
        rotor_phase_angle: float,
        phase_divergence: float,
        invariants: List[str],
        confidence: float
    ) -> ProvenanceStep:
        step = ProvenanceStep(
            step_id=len(self.history) + 1,
            thesis_wave_magnitude=wave_magnitude,
            antithesis_internal_tension=internal_tension,
            synthesis_calibrated_potential=calibrated_potential,
            rotor_phase_angle=rotor_phase_angle,
            phase_divergence=phase_divergence,
            invariants_extracted=invariants,
            validity_confidence=confidence
        )
        self.history.append(step)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return step

    def self_evident_validation_score(self) -> float:
        """
        Calculates overall self-evident validation score across recorded provenance history.
        Grounded in continuous structural consistency rather than static matching.
        """
        if not self.history:
            return 0.0
        scores = [step.validity_confidence for step in self.history]
        return float(np.mean(scores))

    def get_lineage_summary(self) -> List[Dict[str, Any]]:
        """Returns summarized lineage trajectory."""
        return [
            {
                "step": s.step_id,
                "thesis": s.thesis_wave_magnitude,
                "antithesis": s.antithesis_internal_tension,
                "synthesis": s.synthesis_calibrated_potential,
                "phase_divergence": s.phase_divergence,
                "confidence": s.validity_confidence
            }
            for s in self.history
        ]


class MinimalCausalOperator:
    """
    Minimal Causal Operator (최소 인과 연산자).
    Calculates tension convergence/divergence and extracts topological invariants
    from raw physical reality waves without brute-force parameter memorization.
    """

    def compute_tension_dynamics(
        self,
        wave: RealityFrictionWave,
        current_tension: float,
        rotor_phase: float
    ) -> Tuple[float, float, float, List[TopologicalInvariant]]:
        """
        Computes (tension_convergence, tension_divergence, phase_angle_delta, invariants).
        """
        # Friction gradient energy
        friction_energy = (
            abs(wave.fluid_potential_diff) * 0.3 +
            abs(wave.boundary_tension) * 0.4 +
            abs(wave.shear_stress) * 0.2 +
            abs(wave.collision_momentum) * 0.1
        )

        # Tension convergence and divergence loops
        wave_phase = math.atan2(wave.boundary_tension, wave.fluid_potential_diff + 1e-6)
        phase_divergence = abs((wave_phase - rotor_phase) % (2 * math.pi))

        tension_convergence = math.tanh(friction_energy / max(current_tension, 1e-5)) * math.cos(phase_divergence)
        tension_divergence = math.sinh(friction_energy * 0.1) * (1.0 - math.cos(phase_divergence))

        # Rotor phase shift (Delta Theta)
        phase_angle_delta = 0.1 * math.sin(phase_divergence) * math.tanh(friction_energy)

        # Extraction of topological invariant tensor structures
        topological_invariants = []

        # 1. Boundary Continuity Invariant
        if abs(wave.boundary_tension) > 0.5:
            topological_invariants.append(
                TopologicalInvariant(
                    name="boundary_continuity",
                    magnitude=float(wave.boundary_tension),
                    topological_dimension=1,
                    is_stable=abs(wave.boundary_tension) < 2.5,
                    curvature_index=math.tanh(wave.boundary_tension)
                )
            )

        # 2. Field Potential Gradient Invariant
        if abs(wave.fluid_potential_diff) > 0.3:
            topological_invariants.append(
                TopologicalInvariant(
                    name="field_potential_gradient",
                    magnitude=float(wave.fluid_potential_diff),
                    topological_dimension=2,
                    is_stable=True,
                    curvature_index=math.sin(wave.fluid_potential_diff)
                )
            )

        # 3. Shear Momentum Coupling Invariant
        if wave.shear_stress * wave.collision_momentum > 0.2:
            topological_invariants.append(
                TopologicalInvariant(
                    name="shear_momentum_coupling",
                    magnitude=float(wave.shear_stress * wave.collision_momentum),
                    topological_dimension=3,
                    is_stable=True,
                    curvature_index=float(np.exp(-abs(wave.shear_stress)))
                )
            )

        return tension_convergence, tension_divergence, phase_angle_delta, topological_invariants


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
    topological_invariants: List[TopologicalInvariant] = field(default_factory=list)
    tension_convergence: float = 0.0
    tension_divergence: float = 0.0
    provenance_step: Optional[ProvenanceStep] = None


class CausalAdaptationLens(CognitiveLens):
    """
    Cognitive lens designed to process physical reality friction and adapt internal state
    through Minimal Causal Operators and Causal Lineage (인과적 족보) tracking.
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

        self.causal_operator = MinimalCausalOperator()
        self.provenance_trajectory = CausalProvenanceTrajectory()

    def refract_reality_wave(self, wave: RealityFrictionWave) -> DiscernmentResult:
        """
        Processes an incoming reality friction wave, discerning invariant causality from variants,
        executing minimal causal operator dynamics, and recording causal lineage.
        """
        # Calculate raw friction magnitude and phase divergence
        raw_friction = (
            abs(wave.fluid_potential_diff) * 0.3 +
            abs(wave.boundary_tension) * 0.4 +
            abs(wave.shear_stress) * 0.2 +
            abs(wave.collision_momentum) * 0.1
        )

        wave_phase = math.atan2(wave.boundary_tension, wave.fluid_potential_diff + 1e-6)
        phase_divergence = abs((wave_phase - self.rotor_phase_angle) % (2 * math.pi))

        # Dynamic Minimal Causal Operator Execution
        t_conv, t_div, phase_angle_delta, topo_invariants = self.causal_operator.compute_tension_dynamics(
            wave=wave,
            current_tension=self.current_tension,
            rotor_phase=self.rotor_phase_angle
        )

        # Dynamic tension adjustment
        tension_gradient = math.tanh(raw_friction / max(self.current_tension, 1e-5))
        new_tension = self.current_tension * (1.0 + 0.2 * tension_gradient + 0.05 * t_conv)
        self.current_tension = float(np.clip(new_tension, self.min_tension, self.max_tension))

        # Discerning Invariants vs Variants
        invariants = [inv.name for inv in topo_invariants]
        variants = {}

        if "boundary_continuity" not in invariants:
            variants["boundary_fluctuation"] = float(wave.boundary_tension)

        if "field_potential_gradient" not in invariants:
            variants["potential_noise"] = float(wave.fluid_potential_diff)

        if "shear_momentum_coupling" not in invariants:
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

        # Update rotor phase angle
        self.rotor_phase_angle = (self.rotor_phase_angle + phase_angle_delta) % (2 * math.pi)

        # Internal potential field self-calibration
        calibrated_potential = self.current_tension * math.cos(self.rotor_phase_angle)
        self.internal_potential_field = calibrated_potential

        # Confidence score based on invariant stability
        confidence = float(np.mean([inv.validity_score() for inv in topo_invariants])) if topo_invariants else 0.2

        # Record Dialectical Provenance Step
        provenance_step = self.provenance_trajectory.record_step(
            wave_magnitude=raw_friction,
            internal_tension=self.current_tension,
            calibrated_potential=calibrated_potential,
            rotor_phase_angle=self.rotor_phase_angle,
            phase_divergence=phase_divergence,
            invariants=invariants,
            confidence=confidence
        )

        return DiscernmentResult(
            friction_magnitude=raw_friction,
            phase_divergence=phase_divergence,
            causal_invariants=invariants,
            causal_variants=variants,
            self_evident_label=label,
            phase_angle_delta=phase_angle_delta,
            calibrated_potential=calibrated_potential,
            topological_invariants=topo_invariants,
            tension_convergence=t_conv,
            tension_divergence=t_div,
            provenance_step=provenance_step
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
            "causal_variants": discernment.causal_variants,
            "provenance_validation_score": self.provenance_trajectory.self_evident_validation_score()
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
        Performs self-alignment of internal cognitive curvatures based on reality friction and provenance validation.
        """
        validation_factor = discernment.provenance_step.validity_confidence if discernment.provenance_step else 0.5
        curvature_delta = 0.05 * math.tanh(discernment.friction_magnitude) * validation_factor
        for dim in ContextualDimension:
            current_curvature = self.lens_engine.lenses[dim].curvature
            new_curvature = float(np.clip(current_curvature + curvature_delta, 0.1, 5.0))
            self.lens_engine.adjust_lens_curvature(dim, new_curvature)

    def discern_structural_necessity(self, wave: RealityFrictionWave) -> Tuple[bool, float, List[TopologicalInvariant]]:
        """
        Actively discerns whether incoming reality friction matches true structural necessity
        versus ephemeral noise by evaluating topological invariants and phase stability.
        """
        discernment = self.adaptation_lens.refract_reality_wave(wave)
        has_invariants = len(discernment.topological_invariants) > 0
        validity = discernment.provenance_step.validity_confidence if discernment.provenance_step else 0.0
        is_structural_necessity = has_invariants and (validity > 0.3)
        return is_structural_necessity, validity, discernment.topological_invariants

    def get_provenance_lineage(self) -> List[Dict[str, Any]]:
        """Retrieves full recorded causal lineage trajectory history."""
        return self.adaptation_lens.provenance_trajectory.get_lineage_summary()
