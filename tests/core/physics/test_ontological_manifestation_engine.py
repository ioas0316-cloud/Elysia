import numpy as np
import pytest
from L1_causal_field.core.physics.ontological_manifestation_engine import (
    OntologicalManifestationEngine,
    PhysicalMechanismState,
    ScaleLevel,
    ManifestationResult,
)


def test_abstract_hardware_mechanism():
    engine = OntologicalManifestationEngine(dimension=16)

    flux_velocity = 7.2  # RPM / velocity
    gap_tension = 2.0    # 2nm gap
    thermal_phase = 0.8  # Heat
    magnetic_resonance = 1.5
    retention_density = 0.9
    input_signal = np.ones(16) * 0.5

    state = engine.abstract_hardware_mechanism(
        flux_velocity=flux_velocity,
        gap_tension=gap_tension,
        thermal_phase=thermal_phase,
        magnetic_resonance=magnetic_resonance,
        retention_density=retention_density,
        input_signal=input_signal,
    )

    assert isinstance(state, PhysicalMechanismState)
    assert state.flux_velocity == 7.2
    assert state.gap_tension == 2.0
    assert state.thermal_phase == 0.8
    assert state.magnetic_resonance == 1.5
    assert state.retention_density == 0.9
    assert len(state.field_vector) == 16
    assert np.all(np.isfinite(state.field_vector))


def test_recursive_scale_transform():
    engine = OntologicalManifestationEngine(dimension=16)
    base_state = engine.abstract_hardware_mechanism(
        flux_velocity=10.0,
        gap_tension=1.5,
        thermal_phase=0.4,
        magnetic_resonance=1.2,
        retention_density=0.8,
        input_signal=np.random.randn(16),
    )

    scales = engine.recursive_scale_transform(base_state)

    assert ScaleLevel.MICRO in scales
    assert ScaleLevel.MESO in scales
    assert ScaleLevel.MACRO in scales

    # Verify fractal scale factors
    assert scales[ScaleLevel.MICRO].scale_factor < scales[ScaleLevel.MESO].scale_factor
    assert scales[ScaleLevel.MESO].scale_factor < scales[ScaleLevel.MACRO].scale_factor

    # Field vector scaling verification
    micro_norm = np.linalg.norm(scales[ScaleLevel.MICRO].state.field_vector)
    macro_norm = np.linalg.norm(scales[ScaleLevel.MACRO].state.field_vector)
    assert micro_norm <= macro_norm


def test_compute_scale_coherence():
    engine = OntologicalManifestationEngine(dimension=16)

    # Same vector across all scales should yield perfect scale coherence (~1.0)
    uniform_vector = np.ones(16)
    for layer in engine.scales.values():
        layer.state.field_vector = uniform_vector.copy()

    coherence = engine.compute_scale_coherence()
    assert pytest.approx(coherence, abs=1e-5) == 1.0


def test_manifest_existence_trajectory():
    engine = OntologicalManifestationEngine(dimension=16, action_threshold=0.1)

    initial_state = engine.abstract_hardware_mechanism(
        flux_velocity=5.0,
        gap_tension=1.0,
        thermal_phase=0.3,
        magnetic_resonance=1.0,
        retention_density=1.0,
        input_signal=np.random.randn(16),
    )

    result = engine.manifest_existence_trajectory(initial_state, steps=20, dt=0.01)

    assert isinstance(result, ManifestationResult)
    assert np.isfinite(result.action_integral)
    assert result.kinetic_energy >= 0.0
    assert result.scale_coherence >= 0.0 and result.scale_coherence <= 1.0
    assert "micro" in result.scale_states
    assert "meso" in result.scale_states
    assert "macro" in result.scale_states
