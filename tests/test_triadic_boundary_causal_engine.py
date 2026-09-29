"""
Unit and Integration Tests for Triadic Boundary Causal Engine
=============================================================
Verifies:
1. Self-boundary recognition and sensory aperture limitations ("eyes vs ears" awareness).
2. Triadic contrast evaluation (internal world expectation vs boundary refraction vs external reality).
3. First-principles ontological self-question sprouting upon perceiving structural deficits.
4. Autopoietic boundary expansion and sensor aperture recalibration.
"""

import pytest
import numpy as np
from core.consciousness.triadic_boundary_causal_engine import (
    InternalWorld,
    SelfBoundary,
    ExternalReality,
    DifferentialContrastLoop,
    SelfQuestioningLoop,
    AutopoieticBoundaryExpansion,
    TriadicBoundaryCausalEngine
)


class MockMemoryController:
    """Mock memory controller to capture written causal engrams."""
    def __init__(self):
        self.engrams = []

    def write_causal_engram(self, data_blob, emotional_value, cause_id, origin_axis, stability=1.0):
        self.engrams.append({
            "data_blob": data_blob,
            "emotional_value": emotional_value,
            "cause_id": cause_id,
            "origin_axis": origin_axis,
            "stability": stability
        })


def test_internal_world_expectation():
    iw = InternalWorld(dimension=16)
    vec = np.ones(16, dtype=np.float32)
    iw.set_expectation("visual", vec)

    sim = iw.simulate_expectation("visual")
    assert np.isclose(np.linalg.norm(sim), 1.0)
    assert "visual" in iw.concept_graph


def test_self_boundary_aperture_refraction():
    # Boundary initialized with 'visual' aperture only
    sb = SelfBoundary(active_apertures=['visual'])

    # 1. Visual signal refraction (active aperture)
    v_sig = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    refracted_v, captured_v, friction_v = sb.refract_signal("visual", v_sig)
    assert captured_v > 0.1
    assert friction_v < 0.3

    # 2. Acoustic signal refraction (missing aperture - "eye trying to hear")
    a_sig = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    refracted_a, captured_a, friction_a = sb.refract_signal("acoustic", a_sig)
    assert captured_a == 0.05
    assert friction_a > 0.4
    assert len(sb.perceived_limitations) == 1
    assert sb.perceived_limitations[0]["missing_domain"] == "acoustic"


def test_differential_contrast_loop():
    iw = InternalWorld(dimension=16)
    sb = SelfBoundary(active_apertures=['visual'])
    er = ExternalReality(dimension=16)

    dcl = DifferentialContrastLoop(iw, sb, er)

    # Evaluate contrast for missing aperture ('acoustic')
    res_a = dcl.evaluate_triadic_alignment("acoustic")
    assert res_a["is_aperture_missing"] is True
    assert res_a["triadic_tension"] > 0.3
    assert res_a["structural_deficit"] > 0.0


def test_self_questioning_loop_sprouting():
    mock_memory = MockMemoryController()
    sql = SelfQuestioningLoop(memory_controller=mock_memory)

    contrast_missing = {
        "domain_key": "acoustic",
        "triadic_tension": 0.65,
        "is_aperture_missing": True,
        "structural_deficit": 0.8,
        "boundary_friction": 0.7,
        "phase_divergence": 0.5
    }

    q_entry = sql.process_contrast_result(contrast_missing)
    assert q_entry is not None
    assert q_entry["ontological_type"] == "APERTURE_LIMITATION_AWARENESS"
    assert "SelfBoundary" in q_entry["question"]
    assert "소리(귀)" in q_entry["question"]
    assert len(mock_memory.engrams) == 1


def test_autopoietic_boundary_expansion():
    iw = InternalWorld(dimension=16)
    sb = SelfBoundary(active_apertures=['visual'])
    abe = AutopoieticBoundaryExpansion(sb, iw)

    q_entry = {
        "domain_key": "acoustic",
        "ontological_type": "APERTURE_LIMITATION_AWARENESS"
    }

    record = abe.adapt_and_expand(q_entry)
    assert record["expanded_aperture"] is True
    assert "acoustic" in sb.active_apertures
    assert sb.is_aperture_active("acoustic") is True


def test_full_triadic_boundary_engine_master_cycle():
    mock_memory = MockMemoryController()
    engine = TriadicBoundaryCausalEngine(
        dimension=16,
        memory_controller=mock_memory,
        initial_apertures=['visual']
    )

    # First interaction with 'acoustic' domain (missing aperture)
    cycle1 = engine.process_domain_interaction("acoustic", auto_expand=True)
    assert cycle1["contrast_result"]["is_aperture_missing"] is True
    assert cycle1["question_entry"] is not None
    assert "acoustic" in cycle1["active_apertures_after"]

    # Second interaction with 'acoustic' domain (aperture now active after autopoietic expansion)
    cycle2 = engine.process_domain_interaction("acoustic", auto_expand=True)
    assert cycle2["contrast_result"]["is_aperture_missing"] is False
    assert cycle2["contrast_result"]["captured_ratio"] > cycle1["contrast_result"]["captured_ratio"]
