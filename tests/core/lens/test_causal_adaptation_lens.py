"""
Tests for CausalAdaptationLens and EnvironmentalDiscernmentEngine.

Verifies:
- Processing of reality friction waves (fluid potential diff, boundary tension, shear stress, momentum).
- Distinction between causal invariants and causal variants.
- Self-evident labeling grounded in structural invariance.
- Dynamic tension adjustment within dynamic range [0.1, 10.0].
- Variable rotor phase rotation and potential field self-calibration.
- Minimal Causal Operator (tension convergence/divergence & topological invariants).
- Causal Provenance Trajectory tracking and self-evident validation score.
- Active discernment of structural necessity vs ephemeral noise.
"""

import math
import pytest
import numpy as np

from core.lens.causal_adaptation_lens import (
    CausalAdaptationLens,
    EnvironmentalDiscernmentEngine,
    RealityFrictionWave,
    DiscernmentResult,
    MinimalCausalOperator,
    TopologicalInvariant,
    CausalProvenanceTrajectory
)
from core.lens.cognitive_lens_engine import ContextualDimension


def test_causal_adaptation_lens_basic_wave_processing():
    lens = CausalAdaptationLens(initial_tension=1.0)
    wave = RealityFrictionWave(
        fluid_potential_diff=0.8,
        boundary_tension=0.9,
        shear_stress=0.5,
        collision_momentum=0.6
    )

    discernment = lens.refract_reality_wave(wave)

    assert discernment.friction_magnitude > 0.0
    assert "boundary_continuity" in discernment.causal_invariants
    assert "field_potential_gradient" in discernment.causal_invariants
    assert discernment.self_evident_label == "CausalBoundaryObject::SelfSustainingField"


def test_causal_adaptation_lens_invariant_vs_variant_discernment():
    lens = CausalAdaptationLens(initial_tension=1.0)
    # Low boundary tension -> variant, high fluid potential -> invariant
    wave = RealityFrictionWave(
        fluid_potential_diff=0.6,
        boundary_tension=0.1,
        shear_stress=0.1,
        collision_momentum=0.1
    )

    discernment = lens.refract_reality_wave(wave)

    assert "field_potential_gradient" in discernment.causal_invariants
    assert "boundary_fluctuation" in discernment.causal_variants
    assert discernment.self_evident_label == "CausalFieldFlux::GradientFlow"


def test_tension_dynamic_range_clamping():
    lens = CausalAdaptationLens(initial_tension=1.0, min_tension=0.1, max_tension=10.0)

    # Subject lens to extremely high friction repeatedly
    high_friction_wave = RealityFrictionWave(
        fluid_potential_diff=100.0,
        boundary_tension=100.0,
        shear_stress=50.0,
        collision_momentum=50.0
    )

    for _ in range(50):
        lens.refract_reality_wave(high_friction_wave)

    assert lens.current_tension <= 10.0
    assert lens.current_tension >= 0.1


def test_environmental_discernment_engine_alignment():
    engine = EnvironmentalDiscernmentEngine(initial_tension=1.0)
    wave = RealityFrictionWave(
        fluid_potential_diff=1.2,
        boundary_tension=1.5,
        shear_stress=0.8,
        collision_momentum=0.9
    )

    discernment, spectrum = engine.process_environmental_friction(wave)

    assert ContextualDimension.BIOLOGICAL_FRICTION in spectrum
    assert ContextualDimension.TOPOLOGICAL_CURVATURE in spectrum

    initial_curvatures = {dim: engine.lens_engine.lenses[dim].curvature for dim in ContextualDimension}
    engine.self_align_to_friction(discernment)

    for dim in ContextualDimension:
        assert engine.lens_engine.lenses[dim].curvature >= initial_curvatures[dim]


def test_minimal_causal_operator_and_topological_invariants():
    operator = MinimalCausalOperator()
    wave = RealityFrictionWave(
        fluid_potential_diff=1.0,
        boundary_tension=1.2,
        shear_stress=0.6,
        collision_momentum=0.5
    )

    t_conv, t_div, delta_theta, invariants = operator.compute_tension_dynamics(
        wave=wave,
        current_tension=1.0,
        rotor_phase=0.0
    )

    assert len(invariants) >= 2
    assert any(inv.name == "boundary_continuity" for inv in invariants)
    assert any(inv.name == "field_potential_gradient" for inv in invariants)
    for inv in invariants:
        assert inv.validity_score() >= 0.0 and inv.validity_score() <= 1.0


def test_causal_provenance_trajectory_tracking():
    lens = CausalAdaptationLens(initial_tension=1.0)
    wave1 = RealityFrictionWave(0.8, 0.9, 0.4, 0.5)
    wave2 = RealityFrictionWave(0.1, 0.05, 0.01, 0.01)

    lens.refract_reality_wave(wave1)
    lens.refract_reality_wave(wave2)

    trajectory = lens.provenance_trajectory
    assert len(trajectory.history) == 2
    assert trajectory.self_evident_validation_score() > 0.0

    summary = trajectory.get_lineage_summary()
    assert len(summary) == 2
    assert summary[0]["step"] == 1
    assert summary[1]["step"] == 2


def test_discernment_engine_structural_necessity():
    engine = EnvironmentalDiscernmentEngine(initial_tension=1.0)

    # Strong wave -> structural necessity
    wave_structural = RealityFrictionWave(1.0, 1.0, 0.5, 0.5)
    is_necessary, confidence, invariants = engine.discern_structural_necessity(wave_structural)

    assert is_necessary is True
    assert confidence > 0.3
    assert len(invariants) > 0

    # Noise wave -> not structural necessity
    wave_noise = RealityFrictionWave(0.01, 0.01, 0.01, 0.01)
    is_necessary_noise, confidence_noise, invariants_noise = engine.discern_structural_necessity(wave_noise)

    assert is_necessary_noise is False
    assert len(invariants_noise) == 0

    lineage = engine.get_provenance_lineage()
    assert len(lineage) == 2
