"""
Unit tests for Waveform Scale Protocol Engine (core/physics/waveform_scale_protocol.py).
"""

import pytest
import torch
from core.physics.waveform_scale_protocol import (
    WaveformScaleProtocolEngine,
    MicroScaleCluster,
    PhaseLockingMetrics,
)


def test_engine_initialization():
    engine = WaveformScaleProtocolEngine(num_micro_nodes=500, macro_dim=32)
    assert engine.num_micro_nodes == 500
    assert engine.macro_dim == 32
    assert engine.micro_cluster.phases.shape[0] == 500
    assert engine.micro_cluster.chromatic_signatures.shape == (500, 3)


def test_order_parameter_computation():
    engine = WaveformScaleProtocolEngine(num_micro_nodes=100)
    # Set all phases equal to zero -> order parameter R should be 1.0
    engine.micro_cluster.phases.zero_()
    R, Psi = engine.compute_order_parameter()
    assert torch.isclose(R, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(Psi, torch.tensor(0.0), atol=1e-5)


def test_spontaneous_phase_locking():
    engine = WaveformScaleProtocolEngine(num_micro_nodes=300, coupling_strength_K=10.0, dt=0.05)

    # Set macro phase
    target_phase = 1.57 # ~pi/2
    engine.set_macro_phase(target_phase, tension=0.9)

    # Initial Order Parameter
    R_init, _ = engine.compute_order_parameter()

    # Step phase synchronization for 30 steps
    for _ in range(30):
        metrics = engine.step_phase_synchronization()

    assert metrics.order_parameter_R > R_init.item()
    assert metrics.order_parameter_R > 0.8
    assert metrics.bottleneck_reduction_ratio > 80.0
    # Chromatic balance order score (Blue) should dominate
    assert metrics.chromatic_balance[1] > metrics.chromatic_balance[2]


def test_scale_lookup_and_forward():
    engine = WaveformScaleProtocolEngine(num_micro_nodes=200)
    res_vec, metrics = engine(input_val=99.0, steps=10)

    assert res_vec.shape[0] == engine.macro_dim
    assert isinstance(metrics, PhaseLockingMetrics)
    assert metrics.order_parameter_R > 0.5
