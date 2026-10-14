"""
Unit tests for Embodied Sensory Arithmetic & Retrocausal Tracing Engine.
Verifies sensory unit registration, noise rectification, chromatic spectrum mapping,
physical phase transformations, symbolic abstraction, and 100% retrocausal back-tracing.
"""

import numpy as np
import pytest
from core.sensory.embodied_sensory_arithmetic import (
    EmbodiedSensoryArithmeticEngine,
    SensoryUnit,
    CausalSequenceStep,
    AbstractedPattern,
    RetrocausalTracingResult
)


@pytest.fixture
def engine():
    return EmbodiedSensoryArithmeticEngine(noise_threshold=0.1, resolution=128)


def test_sensory_unit_registration_and_rectification(engine):
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    raw_wave = np.sin(t) + 0.05 * np.random.randn(128).astype(np.float32)

    unit = engine.register_sensory_unit("unit_1", raw_wave, amplitude=1.0)

    assert isinstance(unit, SensoryUnit)
    assert unit.unit_id == "unit_1"
    assert unit.rectified is True
    assert len(unit.chromatic_vector) == 3
    assert np.all(unit.chromatic_vector >= 0.0)


def test_physical_transformation_addition(engine):
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    wave_1 = np.sin(t)
    wave_2 = np.cos(t)

    engine.register_sensory_unit("u1", wave_1, amplitude=1.0)
    engine.register_sensory_unit("u2", wave_2, amplitude=1.0)

    step = engine.execute_physical_transformation("ADDITION", ["u1", "u2"], timestamp=1.0)

    assert isinstance(step, CausalSequenceStep)
    assert step.step_index == 1
    assert step.operation_type == "ADDITION"
    assert step.input_unit_ids == ["u1", "u2"]
    assert step.energy_transferred > 0.0


def test_symbolic_mapping_abstraction(engine):
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    wave = np.sin(2 * t)

    engine.register_sensory_unit("obj_1", wave, amplitude=1.0)
    engine.register_sensory_unit("obj_2", wave, amplitude=1.0)
    engine.execute_physical_transformation("ADDITION", ["obj_1", "obj_2"])

    pattern = engine.extract_symbolic_mapping()

    assert isinstance(pattern, AbstractedPattern)
    assert pattern.perceived_quantity > 0.0
    assert pattern.harmony_coherence > 0.0
    assert "Quantity(" in pattern.symbolic_representation


def test_retrocausal_trajectory_back_tracing(engine):
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    wave = np.sin(t)

    engine.register_sensory_unit("s1", wave, amplitude=1.0)
    engine.register_sensory_unit("s2", wave, amplitude=1.0)
    engine.execute_physical_transformation("ADDITION", ["s1", "s2"])

    retro = engine.trace_retrocausal_trajectory()

    assert isinstance(retro, RetrocausalTracingResult)
    assert retro.reconstruction_fidelity > 0.9  # High fidelity match
    assert len(retro.reconstructed_initial_units) == 2
    assert len(retro.reconstructed_sequence) == 1
    assert retro.reversibility_dissipation_index >= 0.0
