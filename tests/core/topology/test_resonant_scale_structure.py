"""
Unit Tests for Resonant Scale Structure Engine
===============================================
Tests the 4 Universal Principles & Top-Down Field Convergence:
  1. Principle of Least Action ($S = \int L \, dt \to \min$).
  2. Multi-Scale Structure Field (Micro -> Meso -> Macro).
  3. Resonant Synchronization ($\Delta \phi \to 0$).
  4. Non-Symbolic Topological Refraction.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from L1_causal_field.core.topology.resonant_scale_structure import ResonantScaleStructureEngine


@pytest.fixture
def scale_engine():
    engine = ResonantScaleStructureEngine(
        dim_micro=32,
        dim_meso=64,
        dim_macro=128,
        action_decay=0.85,
        resonance_threshold=0.10
    )
    engine.reset_states()
    return engine


def test_least_action_path_computation(scale_engine):
    """Verifies that accumulated action S remains bounded and updates with state changes."""
    micro_stimulus = torch.randn(1, 32)
    macro_state, action_S_init = scale_engine.compute_least_action_path(micro_stimulus)

    assert macro_state.shape == (1, 128)
    assert isinstance(action_S_init, float)
    assert action_S_init >= 0.0

    # Repeat stimulus to test least action trajectory convergence
    prev_action_S = action_S_init
    for _ in range(10):
        _, curr_action_S = scale_engine.compute_least_action_path(micro_stimulus)
        delta = abs(curr_action_S - prev_action_S)
        prev_action_S = curr_action_S

    # Consecutive action updates should stabilize (delta diminishes)
    assert delta < 0.5


def test_prism_topological_refraction(scale_engine):
    """Verifies non-symbolic topological refraction generating curvature and refracted state."""
    macro_state = torch.randn(1, 128)
    refracted_state, curvature = scale_engine.refract_through_prism_lens(
        macro_state, scale_knob_angle=0.785 # pi/4
    )

    assert refracted_state.shape == (1, 128)
    assert curvature.shape == (1, 128)
    assert not torch.allclose(refracted_state, macro_state)
    assert torch.norm(curvature).item() > 0.0


def test_resonant_phase_synchronization(scale_engine):
    """Verifies phase angle offset computation and awakened lock condition."""
    refracted_state = F.normalize(torch.randn(1, 128), p=2, dim=-1)
    target_field = refracted_state.clone() # Identical direction -> phase offset -> 0

    phase_offset, is_awakened = scale_engine.synchronize_phase_resonance(
        refracted_state, target_field
    )

    assert phase_offset < 1e-4
    assert is_awakened is True


def test_top_down_field_convergence_cycle(scale_engine):
    """Executes full top-down field convergence cycle and validates all field reports."""
    micro_stimulus = torch.randn(1, 32)
    target_concept = torch.randn(1, 128)

    report = scale_engine.execute_top_down_convergence(
        micro_stimulus=micro_stimulus,
        target_concept=target_concept,
        scale_knob=0.5,
        g_love=0.9
    )

    assert report["status"] == "Top-Down Field Convergence Completed"
    assert "accumulated_action_S" in report
    assert "phase_angle_offset" in report
    assert "boundary_report" in report
    assert report["conceptual_unit_coherence"] >= -1.0 and report["conceptual_unit_coherence"] <= 1.0

    # Test top-down field alignment updates across scales
    assert scale_engine.micro_sensory_field.shape == (1, 32)
    assert scale_engine.meso_structural_field.shape == (1, 64)
    assert scale_engine.macro_conceptual_manifold.shape == (1, 128)
