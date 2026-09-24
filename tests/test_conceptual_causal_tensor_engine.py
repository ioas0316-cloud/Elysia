"""
Unit tests for ConceptualCausalTensorEngine
============================================
Tests high-dimensional conceptual causal tensor geometry, potentiometer dial adaptation (q_err -> 0),
phase state evaluation (GAS -> LIQUID -> ICE), negative mold restoration into positive relief,
and causal provenance logging.
"""

import pytest
import torch
from core.physics.conceptual_causal_tensor_engine import ConceptualCausalTensorEngine


def test_engine_initialization():
    engine = ConceptualCausalTensorEngine(anchor_dim=8, env_dim=8, causal_dim=4)
    assert engine.A_inv.shape == (8, 4)
    assert engine.W_pot.shape == (8, 8)
    assert engine.phase_tensor.shape == (8, 8)


def test_mirror_discrepancy_and_potentiometer_adaptation():
    torch.manual_seed(42)
    engine = ConceptualCausalTensorEngine(anchor_dim=8, env_dim=8, causal_dim=4, learning_rate=0.1)

    P_env = torch.randn(8, 4)
    initial_q_err, P_mapped = engine.compute_mirror_discrepancy(P_env)

    assert initial_q_err.item() > 0.0

    error_history = engine.adapt_potentiometers(P_env, steps=20)
    final_q_err, _ = engine.compute_mirror_discrepancy(P_env)

    assert final_q_err.item() < initial_q_err.item()
    assert len(error_history) > 0


def test_phase_state_transitions_and_positive_relief():
    torch.manual_seed(42)
    engine = ConceptualCausalTensorEngine(anchor_dim=8, env_dim=8, causal_dim=4, phi_solid=0.5, phi_gas=0.1)

    P_env = torch.randn(8, 4)
    output = engine(P_env, auto_adapt=True, adapt_steps=25)

    assert "initial_state" in output
    assert "final_state" in output
    assert output["final_q_err"] <= output["initial_q_err"]
    assert output["positive_relief"].shape == (8, 4)
    assert len(output["adaptation_trajectory"]) > 0


def test_causal_provenance_logging():
    engine = ConceptualCausalTensorEngine(anchor_dim=8, env_dim=8, causal_dim=4)
    P_env = torch.randn(8, 4)

    engine(P_env, auto_adapt=True, adapt_steps=5)
    provenance = engine.get_causal_provenance()

    assert len(provenance) >= 5
    assert "step" in provenance[0]
    assert "q_err" in provenance[0]
    assert "W_pot_norm" in provenance[0]
