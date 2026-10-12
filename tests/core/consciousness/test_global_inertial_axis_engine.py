r"""
Tests for GlobalInertialAxisEngine
==================================
Verifies:
  1. Engine initialization and state reset functionality.
  2. Local Zero filtering: noise suppression of predictable background vs. salient delta extraction.
  3. Scale ladder cross-scale projection (Micro -> Meso -> Macro Face).
  4. Geometric Inertial Axis & angular momentum dynamics.
  5. Self-introspective reverse mirror diagnosis loop & resistance potentiometer recalibration.
  6. Gradient flow and differentiability across iterations.
"""

import pytest
import torch
import torch.nn as nn
from core.consciousness.global_inertial_axis_engine import GlobalInertialAxisEngine


@pytest.fixture
def engine():
    return GlobalInertialAxisEngine(
        dim_micro=32,
        dim_meso=64,
        dim_macro=128,
        local_zero_alpha=0.8,
        local_zero_threshold=0.15,
        inertia_decay=0.92,
        self_healing_rate=0.05,
    )


def test_initialization_and_reset(engine):
    assert engine.dim_micro == 32
    assert engine.dim_meso == 64
    assert engine.dim_macro == 128

    # Ensure buffer shapes
    assert engine.local_zero_baseline.shape == (1, 32)
    assert engine.resistance_potentiometer.shape == (1, 32)
    assert engine.macro_face_state.shape == (1, 128)
    assert engine.inertial_axis.shape == (1, 128)
    assert engine.angular_momentum.shape == (1, 128)

    # Test reset
    engine.local_zero_baseline.fill_(5.0)
    engine.reset_states()
    assert torch.allclose(engine.local_zero_baseline, torch.zeros(1, 32))
    assert torch.allclose(engine.resistance_potentiometer, torch.ones(1, 32))


def test_local_zero_filtering(engine):
    # Repeat the same background input multiple times to establish Local Zero baseline
    bg_input = torch.ones(1, 32) * 2.0
    for _ in range(25):
        salient_delta = engine.filter_local_zero(bg_input)

    # Baseline should adapt close to 2.0
    assert torch.mean(engine.local_zero_baseline).item() > 1.8

    # With constant background input, salient delta should be suppressed near zero
    salient_delta_bg = engine.filter_local_zero(bg_input)
    assert torch.norm(salient_delta_bg).item() < 0.3

    # A sudden novel pulse should break Local Zero equilibrium and produce high salient delta
    novel_input = torch.ones(1, 32) * 10.0
    salient_delta_novel = engine.filter_local_zero(novel_input)
    assert torch.norm(salient_delta_novel).item() > 1.0


def test_forward_pass_and_scale_ladder(engine):
    tick_input = torch.randn(2, 32)
    output = engine(tick_input)

    assert output["status"] == "Global Inertial Axis Pass Executed"
    assert "tension_magnitude" in output
    assert "coherence_alignment" in output
    assert "macro_face_distortion" in output
    assert "max_traced_micro_strain" in output

    assert output["macro_face_state"].shape == (2, 128)
    assert output["inertial_axis"].shape == (1, 128)
    assert output["traced_micro_strain"].shape == (2, 32)


def test_introspective_reverse_diagnosis(engine):
    initial_potentiometer = engine.resistance_potentiometer.clone()

    # Distort macro face state and execute forward passes
    large_input = torch.randn(1, 32) * 5.0
    for _ in range(5):
        out = engine(large_input)

    # Introspective diagnosis should alter resistance potentiometer and accumulate scars
    assert not torch.allclose(engine.resistance_potentiometer, initial_potentiometer)
    assert torch.max(engine.micro_scar_map).item() > 0.0


def test_gradient_flow(engine):
    tick_input = torch.randn(1, 32, requires_grad=True)
    out = engine(tick_input)

    loss = torch.norm(out["macro_face_state"]) + out["salient_tension_delta"].sum()
    loss.backward()

    assert tick_input.grad is not None
    assert torch.norm(tick_input.grad).item() > 0.0
