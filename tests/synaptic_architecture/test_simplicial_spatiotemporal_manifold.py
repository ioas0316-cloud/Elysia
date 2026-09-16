"""
Test suite for Simplicial Spatiotemporal Manifold Engine
=========================================================

Verifies:
1. 0D atom nodes, 1D trajectories, 2D context plane boundaries (SDF validation).
2. 3D multi-sensory cross-modal phase-locking and inner product coherence.
3. 4D spacetime dial phase velocity transitions (continuous vs. structural phase transition).
4. Persistent Homology topological regularization loss against phase collapse.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from synaptic_architecture.simplicial_spatiotemporal_manifold import (
    AtomNode0D,
    CausalTrajectory1D,
    ContextField2D,
    SimplicialSpatiotemporalManifold,
    SimplicialManifoldPipeline,
)


def test_atom_node_and_trajectory():
    pipeline = SimplicialManifoldPipeline(d_model=32)
    node1 = pipeline.add_atom_node("cho_g", "text_symbol", np.ones(32))
    node2 = pipeline.add_atom_node("jung_a", "text_symbol", np.ones(32) * 2.0)

    assert node1.node_id == "cho_g"
    assert node2.node_id == "jung_a"

    traj = pipeline.add_causal_trajectory("cho_g", "jung_a")
    assert traj.source_id == "cho_g"
    assert traj.target_id == "jung_a"
    assert torch.isclose(torch.norm(traj.transition_vector), torch.tensor(np.sqrt(32), dtype=torch.float32))


def test_hangul_syllable_2d_context_field():
    pipeline = SimplicialManifoldPipeline(d_model=32)
    pipeline.add_atom_node("G", "text_symbol", np.array([1.0] * 32))
    pipeline.add_atom_node("A", "text_symbol", np.array([2.0] * 32))
    pipeline.add_atom_node("M", "text_symbol", np.array([1.5] * 32))

    field = pipeline.create_hangul_syllable_plane("GAM", "G", "A", "M")

    assert field.field_id == "Plane_GAM"
    assert len(field.boundary_nodes) == 3

    # Point at center should be strictly inside SDF boundary
    center = field.sdf_center
    assert field.is_inside_sdf_boundary(center) is True
    assert field.compute_sdf(center).item() < 0.0

    # Point far away should be outside
    far_point = center + 10.0
    assert field.is_inside_sdf_boundary(far_point) is False
    assert field.compute_sdf(far_point).item() > 0.0


def test_cross_modal_3d_manifold_and_homology_loss():
    pipeline = SimplicialManifoldPipeline(d_model=32)

    # Add 0D nodes across modalities
    pipeline.add_atom_node("txt_1", "text_symbol", np.random.randn(32))
    pipeline.add_atom_node("snd_1", "acoustic_sound", np.random.randn(32))
    pipeline.add_atom_node("vis_1", "visual_color", np.random.randn(32))
    pipeline.add_atom_node("tex_1", "physical_texture", np.random.randn(32))

    manifold, coherence, loss = pipeline.process_cross_modal_manifold(
        text_nodes=["txt_1"],
        sound_nodes=["snd_1"],
        visual_nodes=["vis_1"],
        texture_nodes=["tex_1"]
    )

    assert manifold.shape[1] == 32
    assert coherence.shape == (4, 4)
    assert loss.item() >= 0.0


def test_4d_spacetime_phase_dial_velocity():
    manifold_layer = SimplicialSpatiotemporalManifold(d_model=32)
    state = torch.randn(10, 32)

    # 1. Slow dial rotation -> Continuous phase shift
    slow_state, info_slow = manifold_layer.step_4d_spacetime_dial(state, dial_delta=0.2, dt=0.1)
    assert info_slow["is_phase_transition"] is False
    assert info_slow["transition_type"] == "Continuous Phase Shift (연속 변이)"
    assert slow_state.shape == state.shape

    # 2. Rapid dial rotation -> Structural phase transition
    fast_state, info_fast = manifold_layer.step_4d_spacetime_dial(state, dial_delta=2.0, dt=0.1)
    assert info_fast["is_phase_transition"] is True
    assert info_fast["transition_type"] == "Structural Phase Transition (상전이)"
    assert fast_state.shape == state.shape
