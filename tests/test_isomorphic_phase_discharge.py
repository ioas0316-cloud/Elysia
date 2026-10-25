"""
Unit Tests for Isomorphic Phase Discharge Engine
Verifies isomorphic phase dynamics across Atmospheric, Electrical, Neural, and Elysia Cognitive scales.
"""

import pytest
import torch
import math
from L1_causal_field.synaptic_architecture.isomorphic_phase_discharge_engine import (
    IsomorphicScale,
    PhaseState,
    FieldStateSnapshot,
    IsomorphicPhaseDischargeEngine
)


def test_initial_superposition_state():
    engine = IsomorphicPhaseDischargeEngine(num_oscillators=256, scale=IsomorphicScale.ELYSIA_ENGINE)
    snapshot = engine.step(dt=0.01)

    assert snapshot.phase_state == PhaseState.SUPERPOSITION
    assert snapshot.barrier_pressure_E_gate < engine.threshold_E_gate
    assert snapshot.discharged_energy == 0.0


def test_isomorphic_scales_parameterization():
    engine = IsomorphicPhaseDischargeEngine(num_oscillators=128)

    for scale in IsomorphicScale:
        engine.set_scale(scale)
        assert engine.scale == scale
        assert engine.threshold_E_gate > 0.0


def test_polarization_and_discharge_cycle():
    engine = IsomorphicPhaseDischargeEngine(
        num_oscillators=256,
        scale=IsomorphicScale.ATMOSPHERIC,
        threshold_E_gate=1.0,
        coupling_strength_K=3.0,
        dissipation_rate=0.01
    )

    # 1. Inject small perturbation to charge E_gate barrier into POLARIZATION state without overflowing threshold
    engine.inject_perturbation(delta_phi=0.2, gradient_tension=0.1)

    snapshot_pol = engine.step(dt=0.05)
    assert snapshot_pol.barrier_pressure_E_gate > 0.0
    assert snapshot_pol.barrier_pressure_E_gate < engine.threshold_E_gate

    # 2. Inject large perturbation to trigger threshold discharge
    for _ in range(10):
        engine.inject_perturbation(delta_phi=0.5, gradient_tension=0.3)

    discharge_triggered = False
    for _ in range(5):
        snap = engine.step(dt=0.05)
        if snap.discharged_energy > 0.0:
            discharge_triggered = True
            break

    assert discharge_triggered, "Discharge surge should trigger when E_gate reaches threshold!"


def test_kuramoto_order_parameter_coherence():
    engine = IsomorphicPhaseDischargeEngine(num_oscillators=128)
    R_initial, _ = engine.compute_order_parameter()
    assert 0.0 <= R_initial <= 1.0

    # Force all micro-oscillators to exact same phase
    engine.theta.fill_(1.2)
    R_synced, _ = engine.compute_order_parameter()
    assert math.isclose(R_synced, 1.0, abs_tol=1e-4)


def test_engine_reset():
    engine = IsomorphicPhaseDischargeEngine(num_oscillators=128)
    engine.inject_perturbation(delta_phi=0.2, gradient_tension=0.1)
    engine.step(dt=0.1)

    engine.reset()
    assert engine.time_step == 0
    assert float(engine.E_gate.item()) == 0.0
    assert float(engine.discharged_energy.item()) == 0.0
    assert engine.phase_state == PhaseState.SUPERPOSITION
