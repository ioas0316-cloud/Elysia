"""
test_falsification_fever.py — Phase 3.5 Falsification Paradigm Test Suite
========================================================================
Rigorous, hard-hitting pytest suite validating the core physical learning,
Axiom Discovery, and Falsification loop (Experiments A through F).
"""

import os
import pytest
import numpy as np
from core.consciousness.axiom_discovery import CausalSpine, AxiomDiscoveryEngine

def test_experiment_a_deterministic_baseline():
    """
    Experiment A: Identical Input -> Identical State.
    Verifies that processing the exact same physical input wave under the same
    internal context yields matching deterministic results.
    """
    spine = CausalSpine(dimensions=3, learning_rate=0.1)

    # Re-align belief states
    spine.belief_state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    spine.transition_matrix = np.eye(3, dtype=np.float32)

    # Process first time
    obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    spine.predict()
    spine.compute_prediction_error(obs)
    spine.update_belief(obs, neuromodulator_alpha=1.0)
    belief1 = np.copy(spine.belief_state)

    # Reset loop belief to original state and run again
    spine.belief_state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    spine.transition_matrix = np.eye(3, dtype=np.float32)

    spine.predict()
    spine.compute_prediction_error(obs)
    spine.update_belief(obs, neuromodulator_alpha=1.0)
    belief2 = np.copy(spine.belief_state)

    assert np.allclose(belief1, belief2, atol=1e-5)

def test_experiment_b_and_c_real_learning_and_persistence():
    """
    Experiment B: New Experience -> Alters future prediction and behavior (Learning).
    Experiment C: Persistence -> The altered behavioral model is maintained over time.
    """
    spine = CausalSpine(dimensions=3, learning_rate=0.5)
    spine.belief_state = np.zeros(3, dtype=np.float32)
    spine.transition_matrix = np.eye(3, dtype=np.float32)

    # Base state: prediction on a zero wave
    spine.predict()
    initial_predicted = np.copy(spine.predicted_state)

    # Experience: Feed highly active wave multiple times to cause Hebbian shift
    obs_active = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(5):
        spine.predict()
        spine.compute_prediction_error(obs_active)
        spine.update_belief(obs_active, neuromodulator_alpha=1.0)

    # Verify belief & prediction have shifted significantly (Learning - Experiment B)
    shifted_predicted = np.copy(spine.predicted_state)
    assert np.linalg.norm(shifted_predicted - initial_predicted) > 0.1

    # Experiment C (Persistence): Run loop with quiet/inert input and ensure the shift remains
    obs_quiet = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    spine.predict()
    spine.compute_prediction_error(obs_quiet)
    spine.update_belief(obs_quiet, neuromodulator_alpha=0.1)

    persisted_predicted = np.copy(spine.predicted_state)

    # The system shouldn't instantly reset back to zero; prediction still carries the learned bias
    assert np.linalg.norm(persisted_predicted - initial_predicted) > 0.05

def test_experiment_d_and_e_falsification_and_rollback():
    """
    Experiment D & E: False experience followed by counter-evidence results in Falsification
    and State Rollback to a clean checkpoint, discarding delusional beliefs.
    """
    spine = CausalSpine(dimensions=3, learning_rate=0.1)
    axiom_discovery = AxiomDiscoveryEngine(correlation_threshold=0.70, window_size=3, falsification_threshold=0.3)

    # 1. Establish a stable invariant over 3 cycles: dopamine and serotonin perfectly correlate
    for i in range(3):
        da = 0.1 * (i + 1)
        se = 0.1 * (i + 1)

        # Layer B: Spine predict and update
        spine.predict()
        actual_obs = np.array([da, 0.5, 0.5], dtype=np.float32)
        spine.compute_prediction_error(actual_obs)
        spine.update_belief(actual_obs)

        # Layer C: Record variables
        live_vars = {
            "dopamine": da,
            "serotonin": se,
            "belief_0": float(spine.belief_state[0])
        }
        axiom_discovery.record_variables(live_vars)

        # Evaluate and discover
        state_snap = {
            "belief_state_0": float(spine.belief_state[0]),
            "belief_state_1": float(spine.belief_state[1]),
            "belief_state_2": float(spine.belief_state[2])
        }
        axiom_discovery.evaluate_and_discover(i, state_snap)

    # Check that a candidate principle was successfully promoted
    assert len(axiom_discovery.principles) >= 1
    principle_id = list(axiom_discovery.principles.keys())[0]
    principle = axiom_discovery.principles[principle_id]
    assert "dopamine" in principle.name or "serotonin" in principle.name

    # Verify we captured a stable checkpoint
    assert "belief_state_0" in principle.stable_state_checkpoint

    # 2. Feed violating counter-evidence: dopamine goes up, serotonin drops to zero
    for i in range(3):
        da = 0.3 + 0.1 * (i + 1)
        se = 0.3 - 0.1 * (i + 1)
        spine.predict()
        actual_obs = np.array([da, 0.5, 0.5], dtype=np.float32)
        spine.compute_prediction_error(actual_obs)
        spine.update_belief(actual_obs)

        live_vars = {
            "dopamine": da,
            "serotonin": se,
            "belief_0": float(spine.belief_state[0])
        }
        axiom_discovery.record_variables(live_vars)

        state_snap = {
            "belief_state_0": float(spine.belief_state[0]),
            "belief_state_1": float(spine.belief_state[1]),
            "belief_state_2": float(spine.belief_state[2])
        }
        axiom_logs = axiom_discovery.evaluate_and_discover(3 + i, state_snap)
        if axiom_discovery.rollback_triggered:
            break

    # Check that falsification and state rollback was triggered
    assert axiom_discovery.rollback_triggered is True

    # Check that the falsified principle was evicted from active candidate principles
    assert principle_id not in axiom_discovery.principles

def test_experiment_f_epistemic_parameter_rejection():
    """
    Experiment F: The system discovers that its own evaluation parameters or candidate
    principles are faulty and rejects them under extreme persistent tension.
    """
    spine = CausalSpine(dimensions=3, learning_rate=0.1)
    axiom_discovery = AxiomDiscoveryEngine(correlation_threshold=0.70, window_size=3, falsification_threshold=0.3)

    # Setup dopamine & serotonin perfect correlation
    for i in range(3):
        da = 0.2 * i
        se = 0.2 * i
        spine.predict()
        actual_obs = np.array([da, 0.5, 0.5], dtype=np.float32)
        spine.compute_prediction_error(actual_obs)
        spine.update_belief(actual_obs)

        live_vars = {
            "dopamine": da,
            "serotonin": se,
            "belief_0": float(spine.belief_state[0])
        }
        axiom_discovery.record_variables(live_vars)
        axiom_discovery.evaluate_and_discover(i, {})

    assert len(axiom_discovery.principles) > 0
    principle_id = list(axiom_discovery.principles.keys())[0]
    principle = axiom_discovery.principles[principle_id]

    # Drive confidence down slowly with persistent violation to trigger rejection
    for i in range(3):
        da = 0.4 + 0.1 * (i + 1)
        se = 0.4 - 0.1 * (i + 1)
        spine.predict()
        actual_obs = np.array([da, 0.5, 0.5], dtype=np.float32)
        spine.compute_prediction_error(actual_obs)
        spine.update_belief(actual_obs)

        live_vars = {
            "dopamine": da,
            "serotonin": se,
            "belief_0": float(spine.belief_state[0])
        }
        axiom_discovery.record_variables(live_vars)
        axiom_discovery.evaluate_and_discover(3 + i, {})

    # The principle should be falsified and discarded as invalid/faulty
    assert principle_id not in axiom_discovery.principles
    assert principle.name in axiom_discovery.falsified_principles
