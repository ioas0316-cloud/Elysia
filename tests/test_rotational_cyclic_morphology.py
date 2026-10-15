"""
Tests for Rotational Cyclic Morphology Module (`core/physics/rotational_cyclic_morphology.py`)
"""

import pytest
import numpy as np
from core.physics.causal_field import CausalField
from core.physics.rotational_cyclic_morphology import (
    PhasorState,
    RotationalCyclicEngine,
    PhaseMorphologyField,
    EnvelopeAreaBlock,
)


def test_phasor_state_complex_conversion():
    state = PhasorState(amplitude=2.0, phase=np.pi / 2, frequency=1.0)
    z = state.to_complex()
    assert np.isclose(np.real(z), 0.0, atol=1e-5)
    assert np.isclose(np.imag(z), 2.0, atol=1e-5)

    restored = PhasorState.from_complex(z, frequency=1.0)
    assert np.isclose(restored.amplitude, 2.0)
    assert np.isclose(restored.phase, np.pi / 2)


def test_rotational_cyclic_engine_energy_conservation():
    engine = RotationalCyclicEngine(default_frequency=1.0, damping=0.0)
    engine.register_phasor("p1", amplitude=1.0, phase=0.0)
    engine.register_phasor("p2", amplitude=3.0, phase=1.0)

    for _ in range(10):
        diag = engine.step(dt=0.1)
        assert np.isclose(diag["energy_conservation_ratio"], 1.0, atol=1e-6)


def test_rotational_cyclic_engine_restoration_force():
    engine = RotationalCyclicEngine(default_frequency=0.0, damping=0.0)  # static frequency to test restoration
    engine.register_phasor("p1", amplitude=1.0, phase=np.pi)

    # Apply restoration towards target phase 0.0
    engine.apply_restoration_force("p1", target_phase=0.0, coupling_k=1.0, dt=0.5)
    state = engine.phasors["p1"]
    # Phase should have shifted towards 0.0 (less than pi)
    assert state.phase < np.pi


def test_phase_morphology_field_scale_projections():
    field = PhaseMorphologyField()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32)

    # Discrete scale (< 0.5)
    field.set_scale(0.2)
    res_discrete = field.project_waveform_geometry(pts)
    assert res_discrete["morph_type"] == "DISCRETE_CUBE_SIMPLEX_LATTICE"

    # Meso scale (0.5 <= scale <= 2.0)
    field.set_scale(1.0)
    res_meso = field.project_waveform_geometry(pts)
    assert res_meso["morph_type"] == "CONTINUOUS_LINE_CURVE_MORPH"

    # Macro scale (> 2.0)
    field.set_scale(3.0)
    res_macro = field.project_waveform_geometry(pts)
    assert res_macro["morph_type"] == "ROTATIONAL_CIRCLE_SPHERE_CURL"


def test_envelope_area_block_filling():
    block = EnvelopeAreaBlock(block_id="test_block", grid_resolution=(10, 10))
    bit_stream = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], dtype=np.int32)

    res = block.extract_envelope_and_fill(bit_stream)
    assert res["block_id"] == "test_block"
    assert res["filled_area"] > 0
    assert 0.0 <= res["filling_ratio"] <= 2.0
    assert 0.0 <= res["density_match_percent"] <= 100.0


def test_envelope_area_block_causal_field_conversion():
    block = EnvelopeAreaBlock(block_id="cf_block", grid_resolution=(8, 8))
    bit_stream = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
    res = block.extract_envelope_and_fill(bit_stream)

    causal_field = CausalField(dimensions=3)
    voxels = block.convert_to_causal_voxels(res, causal_field)

    assert len(voxels) == int(res["filled_area"])
    assert len(causal_field.voxels) == len(voxels)
    assert len(causal_field.beams) > 0
