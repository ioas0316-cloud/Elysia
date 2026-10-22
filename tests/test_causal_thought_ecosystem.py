"""
tests/test_causal_thought_ecosystem.py

Comprehensive Pytest suite for ELYSIA Causal Thought Ecosystem.
Verifies:
1. Sensory Scale Boundary Gate waveform ingestion and friction calculation.
2. Cognitive friction transmutation into Telos growth energy.
3. Dynamic Horizon Expansion (N -> N+1) upon novel sensory/conceptual dissonance ('bitter' taste waveform).
4. Cellular bifurcation in prismatic lens array.
5. Meta-Observer self-reflection and thought trajectory construction.
"""

import pytest
import numpy as np
from elysia_core.causal_thought_ecosystem import CausalThoughtEcosystem, SensoryScaleBoundaryGate, MetaObserverLoop


def test_sensory_scale_boundary_gate_ingestion():
    gate = SensoryScaleBoundaryGate(dimension=3, constraint_order_strength=0.8)
    optical_waveform = np.array([0.8, 0.2, 0.1])
    res = gate.receive_waveform(optical_waveform, sensory_type="optical_micro_lens")

    assert "boundary_friction" in res
    assert res["sensory_type"] == "optical_micro_lens"
    assert res["boundary_friction"] >= 0.0


def test_causal_thought_ecosystem_cycle_and_horizon_expansion():
    ecosystem = CausalThoughtEcosystem(
        initial_dimension=3,
        expansion_threshold=0.35,
        friction_threshold=0.25,
        initial_axis_labels=["sweet", "sour", "salty"]
    )

    initial_dim = ecosystem.dimension
    assert initial_dim == 3

    # Cycle 1: Known stimulus (Sweet) with crayon mechanism (high friction) -> triggers cellular bifurcation
    sweet_waveform = np.array([0.9, 0.1, 0.0])
    res1 = ecosystem.process_thought_cycle(
        raw_sensory_waveform=sweet_waveform,
        stimulus_label_hint="sweet",
        sensory_type="optical_micro_lens",
        mechanism="crayon"
    )

    assert res1["cycle"] == 1
    assert res1["dimension"] == 3
    assert not res1["horizon_expanded"]
    assert res1["growth_energy"] >= 0.0

    # Cycle 2: Known stimulus (Sour)
    sour_waveform = np.array([0.1, 0.9, 0.0])
    res2 = ecosystem.process_thought_cycle(
        raw_sensory_waveform=sour_waveform,
        stimulus_label_hint="sour",
        sensory_type="acoustic_wave",
        mechanism="pastel"
    )

    assert res2["dimension"] == 3

    # Cycle 3: Completely novel orthogonal stimulus ('bitter' / 쓴맛)
    # Stimulus with strong component along 4th orthogonal axis
    bitter_waveform = np.array([0.0, 0.0, 0.0, 1.0])
    res3 = ecosystem.process_thought_cycle(
        raw_sensory_waveform=bitter_waveform,
        stimulus_label_hint="bitter",
        sensory_type="chemical_gustatory",
        mechanism="light"
    )

    # Should trigger Horizon Expansion 3 -> 4
    assert res3["horizon_expanded"]
    assert res3["dimension"] == 4
    assert ecosystem.dimension == 4
    assert "bitter" in res3["active_axis_labels"]
    assert res3["created_axis_label"] == "bitter"

    # Verify status
    status = ecosystem.get_ecosystem_status()
    assert status["dimension"] == 4
    assert status["thought_cycles"] == 3
    assert status["accumulated_growth_energy"] > 0.0
    assert status["total_thought_trajectories"] == 3
    assert status["bifurcated_cores_count"] >= 1


def test_meta_observer_reflection():
    observer = MetaObserverLoop(dimension=4)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    trajectory = np.array([0.8, 0.1, 0.0, 0.0])

    res = observer.reflect_and_evaluate(
        current_state=state,
        thought_trajectory=trajectory,
        growth_energy=0.5,
        active_axes=["sweet", "sour", "salty", "bitter"]
    )

    assert "meta_validity" in res
    assert res["self_reflective_verdict"] == "LOGICAL_CAUSAL_INTEGRITY_VERIFIED"
