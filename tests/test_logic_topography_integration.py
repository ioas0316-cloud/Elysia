"""
Integration Test Suite: Bio-Organismic Logic Topography Engine
=============================================================
Verifies that Elysia can transduce raw math/code structures as continuous waveforms,
manage homeostasis deficits, handle tension and sleep/wake cycles under hazard/contradiction,
and use Clifford Backpropagation (Causal Retrodiction / PyTorch Autograd) to resolve logical errors.
"""

import pytest
import torch
import numpy as np

from core.physics.logic_topography_engine import (
    LogicTopographyEngine,
    SensoryTransducer,
    InnerVitalityHomeostasis,
    CliffordRotorNetwork
)


def test_sensory_transduction_waveforms():
    """Verifies that mathematical and code strings are correctly transduced into continuous waveforms."""
    transducer = SensoryTransducer(resolution=64)

    # Simple equation
    left, right = transducer.split_equality_parts("a + b = c")
    assert left.shape == (64,)
    assert right.shape == (64,)

    # Verify normalization (max absolute value should be 1.0)
    assert torch.max(torch.abs(left)).item() == pytest.approx(1.0, rel=1e-5)
    assert torch.max(torch.abs(right)).item() == pytest.approx(1.0, rel=1e-5)

    # No equality part
    left_no_eq, right_no_eq = transducer.split_equality_parts("x / 2")
    assert left_no_eq.shape == (64,)
    assert right_no_eq.shape == (64,)
    assert torch.all(right_no_eq == 0.0)


def test_clifford_rotor_sandwich_orthogonalization():
    """Verifies that the CliffordRotorNetwork correctly applies Rodriguez-type sandwich rotation."""
    rotor = CliffordRotorNetwork(d_model=32)
    v_in = torch.randn(32)
    v_in = v_in / torch.norm(v_in)

    v_out = rotor(v_in.unsqueeze(0)).squeeze(0)

    # Output should remain norm-preserved on the unit hyper-sphere
    assert torch.norm(v_out).item() == pytest.approx(1.0, rel=1e-5)


def test_homeostasis_energy_and_sleep_cycle():
    """Verifies that homeostasis responds correctly to high potential energy and undergoes sleep annealing."""
    homeostasis = InnerVitalityHomeostasis()
    assert homeostasis.state == "ACTIVE"

    # Step with 0 potential energy (peace)
    homeostasis.step_metabolism(potential_energy=0.0)
    assert homeostasis.state == "IDLE"

    # Step with high potential energy (exhaustion and chaos)
    while homeostasis.state != "SLEEP":
        homeostasis.step_metabolism(potential_energy=1.2)

    assert homeostasis.state == "SLEEP"
    assert homeostasis.sleep_cycles == 10

    # Step metabolism while in sleep state (annealing reduces deficits)
    initial_order = homeostasis.order
    homeostasis.step_metabolism(potential_energy=0.0)
    assert homeostasis.order < initial_order
    assert homeostasis.sleep_cycles == 9


def test_clifford_backpropagation_causal_debugging():
    """
    Verifies that Clifford Backpropagation (Causal Retrodiction) successfully minimizes
    potential energy for logical equations using PyTorch autograd gradients.
    """
    engine = LogicTopographyEngine(resolution=128)

    # Initial potential on an equation proposal
    report_init = engine.process_logic_stream("x + y = z", lr=0.0) # Zero learning rate means no optimization
    init_pot = report_init["initial_potential"]

    # Let the system optimize the rotor angles (retrodiction debugging)
    # Use a solid learning rate to trigger backprop optimization
    report_opt = engine.process_logic_stream("x + y = z", lr=0.5)
    final_pot = report_opt["final_potential"]

    # Potential energy must decrease after Clifford Backprop tuning
    assert final_pot < init_pot
    assert report_opt["resonance_score"] > report_init["resonance_score"]


def test_logical_contradiction_energy_spikes():
    """Verifies that logical hazards (e.g. division by zero, infinite loop) trigger extreme potential energy spikes."""
    engine = LogicTopographyEngine(resolution=64)

    # Normal math equation
    normal_report = engine.process_logic_stream("1 + 1 = 2")
    normal_pot = normal_report["final_potential"]

    # Division by zero hazard equation
    hazard_report = engine.process_logic_stream("y = 10 / 0")
    hazard_pot = hazard_report["final_potential"]

    # Contradiction/singularity should have massively higher potential energy
    assert hazard_pot > normal_pot + 4.0
    assert hazard_report["homeostasis_tension"] > normal_report["homeostasis_tension"]
