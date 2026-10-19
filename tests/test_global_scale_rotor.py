"""
Tests for Global Scale Rotor & Causal-Cognitive PoC Engine (PoC v0.1)

Validates Hypotheses H1-H6, Experiments A, B, C, O(1) ScaleRotor operations,
Local Rotor Rewind ("Eye of Agamotto"), Rotor Coupling Constraint convergence,
and Recursive Scale Hierarchy compression.
"""

import pytest
import numpy as np
from core.topology.global_scale_rotor import (
    SpectralWaveformTensor,
    ScaleRotor,
    RotorCouplingConstraint,
    RecursiveScaleHierarchy,
    CausalCognitiveMemory,
    ObservationEvent,
    CausalBelief
)


def test_spectral_waveform_tensor_creation():
    tensor = SpectralWaveformTensor(shape=(4, 4))
    assert tensor.shape == (4, 4)
    assert tensor.amplitudes.shape == (4, 4)
    assert tensor.phases.shape == (4, 4)
    assert tensor.frequencies.shape == (4, 4)
    assert tensor.compute_energy() > 0.0


def test_scale_rotor_global_phase_rotation_and_inversion():
    tensor = SpectralWaveformTensor(shape=(4, 4))
    initial_phase = tensor.phases.copy()

    # O(1) Global Phase Rotation
    rotated = ScaleRotor.rotate_global_phase(tensor, phase_shift=np.pi / 2)
    np.testing.assert_allclose(rotated.phases, (initial_phase + np.pi / 2) % (2 * np.pi))

    # O(1) Order Inversion
    inverted = ScaleRotor.invert_global_order(rotated)
    assert inverted.amplitudes.shape == (4, 4)


def test_scale_rotor_local_eye_of_agamotto_rewind():
    tensor = SpectralWaveformTensor(shape=(6, 6))
    roi_mask = np.zeros((6, 6), dtype=bool)
    roi_mask[2:4, 2:4] = True  # Local apple region

    # Apply local phase shift & rewind inside ROI
    local_rewound = ScaleRotor.apply_local_rotor(tensor, roi_mask, phase_shift=np.pi / 4, rewind_factor=1.0)

    # Outside ROI should remain unchanged
    np.testing.assert_allclose(local_rewound.phases[~roi_mask], tensor.phases[~roi_mask])
    # Inside ROI should reflect local phase rewind
    assert not np.array_equal(local_rewound.phases[roi_mask], tensor.phases[roi_mask])


def test_rotor_coupling_constraint_relaxation():
    tensor_a = SpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), np.pi / 2))
    tensor_b = SpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), 0.0))

    coupling = RotorCouplingConstraint(constraint_id="joint_1", target_phase_diff=0.0, coupling_strength=0.8)
    initial_torque = coupling.compute_torque(tensor_a, tensor_b)
    assert abs(initial_torque) > 0.0

    # Step toward equilibrium
    new_a, new_b, residual_error = coupling.relax_step(tensor_a, tensor_b)
    new_torque = coupling.compute_torque(new_a, new_b)
    assert abs(new_torque) < abs(initial_torque)


def test_recursive_scale_hierarchy_compression():
    hierarchy = RecursiveScaleHierarchy(layer_id="master_orchestra_L0", scale_level=0)
    t1 = SpectralWaveformTensor(shape=(4, 4), frequencies=np.full((4, 4), 10.0))
    t2 = SpectralWaveformTensor(shape=(4, 4), frequencies=np.full((4, 4), 20.0))

    hierarchy.register_sub_manifold(t1)
    hierarchy.register_sub_manifold(t2)

    compressed_node = hierarchy.compress_to_node()
    assert compressed_node.shape == (1, 1)
    assert compressed_node.frequencies[0, 0] == 15.0  # Mean frequency


def test_experiment_a_property_change_and_identity_maintenance():
    memory = CausalCognitiveMemory()

    # Observation 1: Red Apple
    obs1 = ObservationEvent(
        event_id="obs_1",
        timestamp=1.0,
        target_id="apple_01",
        attributes={"category": "Apple", "color": "Red", "species": "Fuji", "mass_g": 180}
    )
    memory.record_observation(obs1)

    # Observation 2: Same Apple turned Blue (Local rotor phase shift)
    obs2 = ObservationEvent(
        event_id="obs_2",
        timestamp=2.0,
        target_id="apple_01",
        attributes={"category": "Apple", "color": "Blue", "species": "Fuji", "mass_g": 180}
    )
    memory.record_observation(obs2)

    result = memory.evaluate_identity_and_property_change(
        target_id="apple_01",
        category_invariant_keys=["category", "species"],
        variant_attribute_keys=["color"]
    )

    assert result["identity_maintained"] is True
    assert result["property_changes"]["color"] == ("Red", "Blue")
    assert result["trajectory_count"] == 2


def test_experiment_b_state_change_damage_and_uncertainty_preservation():
    memory = CausalCognitiveMemory()

    # Case B1: Observed damage cause
    obs_b1 = ObservationEvent(
        event_id="obs_b1",
        timestamp=3.0,
        target_id="apple_damaged",
        attributes={"category": "Apple", "damage_level": 0.8, "is_fragmented": True},
        context_conditions={"observed_cause": "Impact by hammer"}
    )
    memory.record_observation(obs_b1)

    eval_b1 = memory.evaluate_state_change_and_classification("apple_damaged")
    assert eval_b1["causal_attribution"] == "OBSERVED_CAUSE"
    assert eval_b1["uncertainty_preserved"] is False

    # Case B2: Ambiguous damage cause (Preserve Uncertainty)
    obs_b2 = ObservationEvent(
        event_id="obs_b2",
        timestamp=3.5,
        target_id="apple_ambiguous",
        attributes={"category": "Apple", "damage_level": 0.9, "is_fragmented": True},
        context_conditions={}  # No observed cause
    )
    memory.record_observation(obs_b2)

    eval_b2 = memory.evaluate_state_change_and_classification("apple_ambiguous")
    assert eval_b2["causal_attribution"] == "INFERRED_UNKNOWN"
    assert eval_b2["uncertainty_preserved"] is True
    assert eval_b2["classification_action"] == "SUSPEND_JUDGMENT"


def test_experiment_c_intertemporal_recurrent_causality():
    memory = CausalCognitiveMemory()

    # Step 1: Initial observation & belief
    obs1 = ObservationEvent(
        event_id="obs_c1",
        timestamp=1.0,
        target_id="subject_x",
        attributes={"category": "Fruit", "state": "Intact", "ripeness": 0.2}
    )
    memory.record_observation(obs1)
    belief1 = memory.update_intertemporal_belief("subject_x", "obs_c1")
    assert belief1.status == "ACTIVE"

    # Step 2: Non-disruptive ripening observation (Maintains ACTIVE status)
    obs2 = ObservationEvent(
        event_id="obs_c2",
        timestamp=2.0,
        target_id="subject_x",
        attributes={"category": "Fruit", "state": "Ripe", "ripeness": 0.9}
    )
    memory.record_observation(obs2)
    belief2 = memory.update_intertemporal_belief("subject_x", "obs_c2")
    assert belief2.status == "ACTIVE"
    assert belief2.attributes["ripeness"] == 0.9
    assert len(belief2.evidence_history) == 2

    # Step 3: Major damage with known cause (Updates belief status to REVISED)
    obs3 = ObservationEvent(
        event_id="obs_c3",
        timestamp=3.0,
        target_id="subject_x",
        attributes={"category": "Fruit", "state": "Smashed", "damage_level": 0.95, "is_fragmented": True},
        context_conditions={"observed_cause": "Crushed by press"}
    )
    memory.record_observation(obs3)
    belief3 = memory.update_intertemporal_belief("subject_x", "obs_c3")
    assert belief3.status == "REVISED"
    assert len(belief3.evidence_history) == 3
