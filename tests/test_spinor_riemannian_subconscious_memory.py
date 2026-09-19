"""
Unit tests for Spatiotemporal Transduction, High-Dimensional Spinor-Riemannian Autograd Layer,
Static Rotor Aux-Memory Unit (SRAM-R), Subconscious SSD Atlas, and Clifford Spinor Recall Engine.
"""

import pytest
import torch

from core.lens.spatiotemporal_transduction_lens import (
    StructuralReceptor,
    TopologicalMapper,
    HighDimSpinorRiemannianLayer,
)
from core.memory.subconscious_rotor_atlas import (
    StaticRotorAuxMemoryUnit,
    SubconsciousSSDAtlas,
    LocalChart,
)
from core.lens.clifford_spinor_recall_engine import (
    CliffordSpinorRecallEngine,
)


def test_transduction_receptor_mapper():
    batch_size = 4
    ext_dim = 32
    bound_dim = 16
    state_dim = 8

    receptor = StructuralReceptor(ext_dim, bound_dim, threshold=0.05)
    mapper = TopologicalMapper(bound_dim, state_dim)

    Xi_ext = torch.randn(batch_size, ext_dim)
    S_bound = receptor(Xi_ext)
    v_shift = mapper(S_bound)

    assert S_bound.shape == (batch_size, bound_dim)
    assert v_shift.shape == (batch_size, state_dim)


def test_high_dim_spinor_riemannian_layer_grad_flow():
    batch_size = 2
    state_dim = 8
    num_anchors = 4

    layer = HighDimSpinorRiemannianLayer(state_dim=state_dim, num_anchors=num_anchors)

    x_init = torch.randn(batch_size, state_dim, requires_grad=True)
    Q_init = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1).requires_grad_(True)
    target = torch.randn(batch_size, state_dim)

    x_next, Q_next = layer(x_init, Q_init, target)

    assert x_next.shape == (batch_size, state_dim)
    assert Q_next.shape == (batch_size, state_dim, state_dim)

    loss = (x_next - target).pow(2).sum() + Q_next.pow(2).sum()
    loss.backward()

    assert x_init.grad is not None
    assert Q_init.grad is not None
    assert layer.anchors.grad is not None
    assert x_init.grad.norm().item() > 0.0


def test_static_rotor_sram_r():
    state_dim = 8
    sram_r = StaticRotorAuxMemoryUnit(state_dim=state_dim)

    tag_id = "tag_test_01"
    Q_curr = torch.eye(state_dim).unsqueeze(0)

    # Freeze Phase
    Omega = sram_r.freeze_phase(tag_id, Q_curr)
    assert tag_id in sram_r.rtr_store
    assert Omega.shape == (1, state_dim, state_dim)

    # Background Drift
    sram_r.update_background_drift(dt=0.01)

    # Sync Phase
    Q_global = torch.eye(state_dim).unsqueeze(0)
    Q_synced = sram_r.sync_phase(tag_id, Q_global)

    assert Q_synced.shape == (1, state_dim, state_dim)


def test_subconscious_ssd_atlas_and_hardening():
    state_dim = 8
    atlas = SubconsciousSSDAtlas(state_dim=state_dim, chart_radius=1.0)

    x_evt = torch.randn(1, state_dim)
    v_shift = torch.randn(state_dim)
    Omega_skew = torch.randn(state_dim, state_dim)
    Omega_skew = 0.5 * (Omega_skew - Omega_skew.T)

    chart = atlas.consolidate_event(x_evt, v_shift, Omega_skew, plasticity_rate=0.05)
    assert chart.lock_count == 1
    assert not chart.is_hardened

    # Harden chart
    for _ in range(5):
        atlas.consolidate_event(x_evt, v_shift, Omega_skew, plasticity_rate=0.05)

    assert chart.is_hardened
    assert chart.gamma == 1e-4

    # Test natural decay step
    g_before = chart.g_mem.clone()
    atlas.step_subconscious_decay(dt=0.1)
    assert chart.g_mem.shape == (state_dim, state_dim)


def test_clifford_spinor_recall_engine():
    state_dim = 8
    engine = CliffordSpinorRecallEngine(state_dim=state_dim)

    x_init = torch.randn(1, state_dim)
    Q_init = torch.eye(state_dim).unsqueeze(0)
    target = torch.randn(1, state_dim)
    g_mem = torch.eye(state_dim) + 0.05 * torch.randn(state_dim, state_dim)
    g_mem = torch.mm(g_mem.T, g_mem)

    Omega = torch.randn(1, state_dim, state_dim)
    Omega = 0.5 * (Omega - Omega.transpose(-1, -2))

    x_traj, Q_traj, energy = engine.recall_dynamics(
        x_init, Q_init, target, g_mem, Omega, steps=10, dt=0.05
    )

    assert x_traj.shape == (11, 1, state_dim)
    assert Q_traj.shape == (11, 1, state_dim, state_dim)
    assert len(energy) == 10
    # Riemannian distance should relax downwards
    assert energy[-1] < energy[0]
