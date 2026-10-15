"""
Unit Tests for Isomorphic Phase Field Substrate & Scale-Hierarchical Protocol
==============================================================================
"""

import numpy as np
import pytest
from core.physics.isomorphic_phase_field import (
    IsomorphicPhaseFieldSubstrate,
    ModalitySignature,
    ScaleBoundaryResult
)


def test_isomorphic_phase_field_initialization():
    """Test phase field initialization and default vacuum state."""
    substrate = IsomorphicPhaseFieldSubstrate(resolution=128, base_frequency=10.0)
    assert len(substrate.grid) == 128
    assert np.allclose(np.abs(substrate.grid), 1.0)


def test_wave_interference_addition():
    """Test wave addition as constructive wave interference & scale expansion."""
    sub1 = IsomorphicPhaseFieldSubstrate(resolution=128)
    sub2 = IsomorphicPhaseFieldSubstrate(resolution=128)

    # Set aligned phasors for maximum constructive interference
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    sub1.set_field_from_phasor(np.exp(1j * t))
    sub2.set_field_from_phasor(np.exp(1j * t))

    combined = sub1.wave_interference_add(sub2)

    power1 = np.sum(np.abs(sub1.grid) ** 2)
    power_combined = np.sum(np.abs(combined.grid) ** 2)

    # Constructive interference doubles amplitude, quadrupling power
    assert power_combined > power1 * 3.5
    assert len(sub1.transformation_history) == 1
    assert sub1.transformation_history[0]["operation"] == "wave_interference_add"


def test_wave_interference_subtraction():
    """Test wave subtraction as destructive wave interference & phase cancellation."""
    sub1 = IsomorphicPhaseFieldSubstrate(resolution=128)
    sub2 = IsomorphicPhaseFieldSubstrate(resolution=128)

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    sub1.set_field_from_phasor(np.exp(1j * t))
    sub2.set_field_from_phasor(np.exp(1j * t))

    rectified = sub1.wave_interference_subtract(sub2)

    power_rectified = np.sum(np.abs(rectified.grid) ** 2)
    # Complete destructive cancellation
    assert power_rectified < 1e-10


def test_autonomous_modality_judgment():
    """Test autonomous modality judgment without external labels/extensions."""
    substrate = IsomorphicPhaseFieldSubstrate(resolution=128)

    # 1. Code / Logic simulation: Sharp phase jumps (high derivative sparsity)
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    code_phasor = np.exp(1j * np.where(t < np.pi, 0.0, np.pi * 0.9))
    substrate.set_field_from_phasor(code_phasor)

    sig_code = substrate.autonomous_modality_judgment()
    assert sig_code.primary_modality == "code_and_logic"
    assert sig_code.confidence_score > 0.3

    # 2. Image / Video simulation: Continuous spatial dispersion
    spatial_phasor = np.exp(1j * np.sin(5 * t) * 3.0)
    substrate.set_field_from_phasor(spatial_phasor)

    sig_img = substrate.autonomous_modality_judgment()
    assert sig_img.primary_modality in ["image_and_video", "geometry_3d"]


def test_scale_boundary_traversal_lossless():
    """Test scale boundary layer traversal reversibility and Delta S ≈ 0."""
    substrate = IsomorphicPhaseFieldSubstrate(resolution=128)
    t = np.linspace(0, 4 * np.pi, 128, endpoint=False)
    substrate.set_field_from_phasor(np.sin(t) * np.exp(1j * t * 0.5))

    res = substrate.traverse_isomorphic_scale_boundary(scale_factor=2.5)

    assert res.reconstructed_fidelity > 0.99
    assert res.reversibility_entropy_delta < 1e-3
    assert res.isomorphic_equivalence is True


def test_multimodal_observation_projections():
    """Test non-destructive multi-modal observation projections."""
    substrate = IsomorphicPhaseFieldSubstrate(resolution=128)

    text_proj = substrate.project_to_text()
    assert "[P_text Narrative]" in text_proj

    code_proj = substrate.project_to_code()
    assert "fn execute_phase_field_step" in code_proj

    img_proj = substrate.project_to_image()
    assert img_proj["axis"] == "P_image"
    assert "hue_phase_field" in img_proj

    geom_proj = substrate.project_to_geometry()
    assert geom_proj["axis"] == "P_geom"
    assert len(geom_proj["point_cloud_3d"]) > 0


def test_macro_rotor_concept_promotion():
    """Test promotion of micro phasors to higher-order Macro-Rotor concept."""
    substrate = IsomorphicPhaseFieldSubstrate(resolution=128)
    macro = substrate.promote_to_macro_rotor_concept("Concept_Causality_1")

    assert macro["concept_id"] == "Concept_Causality_1"
    assert "coherence_index" in macro
    assert macro["scale_level"] == "MacroHierarchy_Level_1"
