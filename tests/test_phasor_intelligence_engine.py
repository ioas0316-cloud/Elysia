"""
Unit Tests for Phase-Field Intelligence Engine (PhasorIntelligenceEngine)
==========================================================================
"""

import numpy as np
import pytest
from core.engine.phasor_intelligence_engine import (
    PhasorIntelligenceEngine,
    PhasorModalitySignature,
    SpatialKinematicLayer
)


def test_phasor_engine_initialization_and_vacuum():
    """Test engine initialization and continuous phasor sea vacuum state."""
    engine = PhasorIntelligenceEngine(resolution=128, base_frequency=10.0)
    assert engine.resolution == 128
    assert len(engine.grid) == 128
    assert np.allclose(np.abs(engine.grid), 1.0)


def test_wave_interference_addition_expansion():
    """Test constructive wave interference addition and power expansion."""
    e1 = PhasorIntelligenceEngine(resolution=128)
    e2 = PhasorIntelligenceEngine(resolution=128)

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    e1.set_field(np.exp(1j * t))
    e2.set_field(np.exp(1j * t))  # Phase aligned

    combined = e1.wave_interference_add(e2)

    power_1 = np.sum(np.abs(e1.grid) ** 2)
    power_combined = np.sum(np.abs(combined.grid) ** 2)

    # In-phase aligned waves double amplitude -> 4x power
    assert power_combined > power_1 * 3.5
    sensation = e1.transformation_history[0]["sensation"]
    assert sensation == "expansion_resonance"


def test_wave_interference_subtraction_rectification():
    """Test destructive wave interference subtraction and phase cancellation."""
    e1 = PhasorIntelligenceEngine(resolution=128)
    e2 = PhasorIntelligenceEngine(resolution=128)

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    e1.set_field(np.exp(1j * t))
    e2.set_field(np.exp(1j * t))

    cancelled = e1.wave_interference_subtract(e2)
    power_cancelled = np.sum(np.abs(cancelled.grid) ** 2)

    assert power_cancelled < 1e-10
    assert e1.transformation_history[0]["sensation"] == "rectification_quietude"


def test_autonomous_modality_judgment():
    """Test autonomous modality judgment without external labels."""
    engine = PhasorIntelligenceEngine(resolution=128)

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    code_phasor = np.exp(1j * np.where(t < np.pi, 0.0, np.pi * 0.9))
    engine.set_field(code_phasor)

    sig = engine.autonomous_modality_judgment()
    assert sig.primary_modality == "code_and_logic"
    assert sig.confidence_score > 0.3


def test_isomorphic_spatial_kinematics_projection():
    """Test multi-layer spatial projection with kinematics (Depth, Width, Optical Flow)."""
    engine = PhasorIntelligenceEngine(resolution=128)
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    engine.set_field(np.exp(1j * np.sin(3 * t) * 2.0))

    spatial_layers = engine.project_to_spatial_kinematics()

    assert len(spatial_layers) == 3
    layer_names = [l.layer_name for l in spatial_layers]
    assert "Background" in layer_names
    assert "Midground" in layer_names
    assert "Foreground" in layer_names

    bg_layer = spatial_layers[0]
    fg_layer = spatial_layers[2]

    # Foreground optical flow should be larger than background optical flow for same field motion
    assert np.mean(np.abs(fg_layer.optical_flow_vectors)) > np.mean(np.abs(bg_layer.optical_flow_vectors))


def test_isomorphic_text_code_geom_projections():
    """Test isomorphic projections across Text, Code, and 3D Geometry axes."""
    engine = PhasorIntelligenceEngine(resolution=128)

    text_out = engine.project_to_text()
    assert "[P_text Stream]" in text_out

    code_out = engine.project_to_code()
    assert "fn process_phasor_field_rotors" in code_out

    geom_out = engine.project_to_geometry_3d()
    assert geom_out["axis"] == "P_geom"
    assert len(geom_out["point_cloud_3d"]) > 0


def test_scale_hierarchical_isomorphic_feedback_loop():
    """Test bi-directional feedback loop between micro-rotors and macro-rotor concepts."""
    engine = PhasorIntelligenceEngine(resolution=128)

    # 1. Micro to Macro promotion
    macro = engine.promote_micro_rotors_to_macro_concept("MacroConcept_Elysia_1")
    assert macro["concept_id"] == "MacroConcept_Elysia_1"
    assert "coherence_index" in macro

    # Perturb micro field with noise
    np.random.seed(42)
    noise_phases = np.random.uniform(-np.pi, np.pi, 128)
    engine.set_field(np.exp(1j * noise_phases))

    coherence_before = np.abs(np.sum(engine.grid)) / np.sum(np.abs(engine.grid))

    # 2. Macro to Micro top-down regulation feedback
    engine.apply_macro_feedback_to_micro_rotors("MacroConcept_Elysia_1", feedback_strength=0.8)

    coherence_after = np.abs(np.sum(engine.grid)) / np.sum(np.abs(engine.grid))

    # Top-down feedback should align phases and significantly increase field coherence
    assert coherence_after > coherence_before
