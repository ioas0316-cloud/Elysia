"""
Tests for ELYSIA v0.2 Clifford Multivector Rotor & Dynamic VRAM Buffer Pruning Engine.
"""

import numpy as np
import pytest
from core.topology.clifford_v02_engine import (
    CliffordMultivector,
    CliffordSpectralWaveformTensor,
    CliffordScaleRotor,
    DynamicBufferPruningEngine,
    CliffordRotorCouplingConstraint,
    CliffordCausalCognitiveMemory,
    CliffordObservationEvent
)


class TestCliffordMultivector:
    def test_rotor_creation_and_sandwich(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2.0  # 90 degrees
        rotor = CliffordMultivector.create_rotor(axis, angle)

        v = np.array([1.0, 0.0, 0.0])
        v_prime = rotor.transform_vector(v)

        # Expected rotation of [1, 0, 0] by 90 deg around Z is [0, 1, 0]
        np.testing.assert_allclose(v_prime, np.array([0.0, 1.0, 0.0]), atol=1e-12)

    def test_counterfactual_rewind_reversibility(self):
        axis = np.array([1.0, 1.0, 0.0])
        angle = 1.234  # Arbitrary angle
        rotor = CliffordMultivector.create_rotor(axis, angle)

        v_batch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.5, -1.0],
            [-3.0, 0.5, 4.0]
        ])

        v_transformed = rotor.transform_vector(v_batch)
        v_restored = rotor.reverse_transform_vector(v_transformed)

        np.testing.assert_allclose(v_batch, v_restored, atol=1e-12)


class TestCliffordSpectralWaveformTensorAndRotors:
    def test_tensor_creation_and_energy(self):
        tensor = CliffordSpectralWaveformTensor(
            shape=(4, 4),
            amplitudes=np.full((4, 4), 2.0),
            frequencies=np.full((4, 4), 10.0)
        )
        # Energy = sum(A^2 * f) = 16 * 4 * 10 = 640
        assert tensor.compute_energy() == pytest.approx(640.0)

    def test_global_and_local_bivector_scale_rotor(self):
        tensor = CliffordSpectralWaveformTensor(shape=(6, 6))
        axis = np.array([0.0, 0.0, 1.0])
        rotor = CliffordMultivector.create_rotor(axis, np.pi / 2.0)

        # Global rotation
        rot_global = CliffordScaleRotor.rotate_global_bivector(tensor, rotor)
        assert np.mean(rot_global.phases) == pytest.approx(np.pi / 2.0, abs=1e-5)

        # Local rewind
        roi_mask = np.zeros((6, 6), dtype=bool)
        roi_mask[2:4, 2:4] = True
        rot_local = CliffordScaleRotor.apply_local_rotor(rot_global, roi_mask, rotor, rewind_factor=1.0)

        assert np.mean(rot_local.phases[roi_mask]) == pytest.approx(0.0, abs=1e-5)
        assert np.mean(rot_local.phases[~roi_mask]) == pytest.approx(np.pi / 2.0, abs=1e-5)


class TestDynamicBufferPruningEngine:
    def test_buffer_pruning_and_hydration(self):
        tensor = CliffordSpectralWaveformTensor(shape=(8, 8))
        tensor.phases[0:4, :] = np.pi  # Decoupled phase

        engine = DynamicBufferPruningEngine(coherence_threshold=0.5)
        pruned = engine.prune_tensor_buffer(tensor, reference_phase=0.0)

        assert pruned.is_pruned is True
        assert np.sum(pruned.pruned_mask) == 32  # Half pruned
        assert np.all(pruned.amplitudes[0:4, :] == 0.0)

        report = DynamicBufferPruningEngine.get_memory_footprint_report(tensor, pruned)
        assert report["reduction_ratio"] == pytest.approx(0.5)

        # Hydration
        hydrated = engine.hydrate_tensor_buffer(pruned)
        assert hydrated.is_pruned is False
        assert np.all(hydrated.amplitudes == 1.0)

    def test_tensor_slicing(self):
        tensor = CliffordSpectralWaveformTensor(shape=(10, 10))
        sliced = DynamicBufferPruningEngine.slice_active_threshold_roi(tensor, (slice(2, 6), slice(2, 6)))

        assert sliced.shape == (4, 4)
        assert sliced.calculate_memory_bytes() < tensor.calculate_memory_bytes()


class TestCliffordRotorCouplingConstraint:
    def test_joint_relaxation_convergence(self):
        t_a = CliffordSpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), np.pi / 2))
        t_b = CliffordSpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), 0.0))

        constraint = CliffordRotorCouplingConstraint("c1", target_phase_diff=0.0, coupling_strength=0.8)

        curr_a, curr_b = t_a, t_b
        for _ in range(10):
            curr_a, curr_b, err = constraint.relax_step(curr_a, curr_b)

        assert err < 0.01


class TestCliffordCausalCognitiveMemory:
    def test_intertemporal_belief_pipeline(self):
        memory = CliffordCausalCognitiveMemory()

        # Phase 1: Normal observation
        obs1 = CliffordObservationEvent(
            event_id="e1", timestamp=1.0, target_id="item_01",
            attributes={"category": "Crystal", "damage_level": 0.0}
        )
        memory.record_observation(obs1)
        belief1 = memory.update_intertemporal_belief("item_01", "e1")
        assert belief1.status == "ACTIVE"

        # Phase 2: Damaged without cause -> SUSPENDED
        obs2 = CliffordObservationEvent(
            event_id="e2", timestamp=2.0, target_id="item_01",
            attributes={"category": "Crystal", "damage_level": 0.9, "is_fragmented": True}
        )
        memory.record_observation(obs2)
        belief2 = memory.update_intertemporal_belief("item_01", "e2")
        assert belief2.status == "SUSPENDED"

        # Phase 3: Cause observed -> REVISED
        obs3 = CliffordObservationEvent(
            event_id="e3", timestamp=3.0, target_id="item_01",
            attributes={"category": "Shards", "damage_level": 0.9, "is_fragmented": True},
            context_conditions={"observed_cause": "High pressure pulse"}
        )
        memory.record_observation(obs3)
        belief3 = memory.update_intertemporal_belief("item_01", "e3")
        assert belief3.status == "REVISED"
        assert belief3.attributes["category"] == "Shards"
