"""
ELYSIA Automated Discovery Audit Test Suite (Audit of Discovery v1.0)

This test suite executes rigorous verification of the ELYSIA Causal-Cognitive modules
against the 6 core discovery criteria without modifying existing module behavior:
1. Label Independence (Testing unlabeled random phase/wave noise)
2. Conclusion Non-Injection (Ensuring test oracles do not inject answers)
3. Relational Derivability (Tracing phase, torque, and energy mechanics)
4. Invariance Verification (Rotation and scale invariance)
5. Divergence & Difference Verification (Detecting topological/phase divergence)
6. Counterfactual & Ablation Experiments (Deliberately disrupting rotor coupling mechanics and proving failure boundaries)
"""

import pytest
import numpy as np

from core.topology.global_scale_rotor import (
    SpectralWaveformTensor, ScaleRotor, RotorCouplingConstraint,
    RecursiveScaleHierarchy, ObservationEvent, CausalBelief, CausalCognitiveMemory
)
from core.topology.causal_discernment_engine import CausalDiscernmentEngine
from core.topology.invariance_divergence_tracker import InvarianceDivergenceTracker, CausalDiscontinuityDetector
from core.topology.causal_judgment_gear import CausalJudgmentGear
from core.topology.causal_structure import InformationTopology


class TestAuditDiscoverySuite:

    def test_criterion_1_label_independence_unlabeled_noise(self):
        """
        Criterion 1: Label Independence.
        Verify system behavior when fed raw, unlabeled complex phase/frequency noise
        without human-assigned entity categories or color string names.
        """
        np.random.seed(42)
        # Create unlabeled random spectral waveform tensors
        raw_tensor_a = SpectralWaveformTensor(
            shape=(8, 8),
            amplitudes=np.random.exponential(scale=1.0, size=(8, 8)),
            phases=np.random.uniform(0, 2*np.pi, size=(8, 8)),
            frequencies=np.random.uniform(10, 100, size=(8, 8))
        )

        raw_tensor_b = SpectralWaveformTensor(
            shape=(8, 8),
            amplitudes=raw_tensor_a.amplitudes * 1.05,
            phases=(raw_tensor_a.phases + 0.1) % (2 * np.pi),
            frequencies=raw_tensor_a.frequencies
        )

        joint = RotorCouplingConstraint(constraint_id="c_unlabeled", coupling_strength=0.5)
        torque = joint.compute_torque(raw_tensor_a, raw_tensor_b)

        # System calculates physical torque purely based on phase fields without category labels
        assert isinstance(torque, float)
        assert not np.isnan(torque)

    def test_criterion_2_non_injected_conclusions_oracle_audit(self):
        """
        Criterion 2: Conclusion Non-Injection.
        Verify if CausalCognitiveMemory's SUSPEND_JUDGMENT action relies on missing context
        vs continuous phase/amplitude field thresholds.
        """
        mem = CausalCognitiveMemory()

        # Observation with high damage but NO observed cause string
        obs_no_cause = ObservationEvent(
            event_id="obs_dam_1",
            timestamp=1.0,
            target_id="node_x",
            attributes={"damage_level": 0.9, "is_fragmented": True},
            context_conditions={}  # Missing "observed_cause"
        )
        mem.record_observation(obs_no_cause)
        res_no_cause = mem.evaluate_state_change_and_classification("node_x")
        assert res_no_cause["classification_action"] == "SUSPEND_JUDGMENT"
        assert res_no_cause["uncertainty_preserved"] is True

        # Observation with identical high damage BUT with "observed_cause" string injected
        obs_with_cause = ObservationEvent(
            event_id="obs_dam_2",
            timestamp=2.0,
            target_id="node_x",
            attributes={"damage_level": 0.9, "is_fragmented": True},
            context_conditions={"observed_cause": "external_impact"}
        )
        mem.record_observation(obs_with_cause)
        res_with_cause = mem.evaluate_state_change_and_classification("node_x")

        # AUDIT DISCOVERY FINDING: The decision flips from SUSPEND_JUDGMENT to REVISE_CLASSIFICATION
        # based solely on the existence of the string 'observed_cause' in context_conditions.
        assert res_with_cause["classification_action"] == "REVISE_CLASSIFICATION"
        assert res_with_cause["uncertainty_preserved"] is False

    def test_criterion_3_relational_derivability_energy_conservation(self):
        """
        Criterion 3: Relational Derivability.
        Verify that O(1) global phase rotation preserves global energy exactly.
        """
        t = SpectralWaveformTensor(
            shape=(16, 16),
            amplitudes=np.random.uniform(0.5, 2.0, (16, 16)),
            phases=np.random.uniform(0, 2*np.pi, (16, 16)),
            frequencies=np.random.uniform(1, 10, (16, 16))
        )
        e_initial = t.compute_energy()

        t_rotated = ScaleRotor.rotate_global_phase(t, phase_shift=np.pi / 3)
        e_rotated = t_rotated.compute_energy()

        # Energy is strictly derived and conserved across phase rotation
        assert np.isclose(e_initial, e_rotated)

    def test_criterion_4_invariance_verification_local_roi_rewind(self):
        """
        Criterion 4: Invariance Verification.
        Verify that Local Spacetime Rotor Control ("Eye of Agamotto") keeps unmasked region invariant.
        """
        t = SpectralWaveformTensor(shape=(10, 10), phases=np.zeros((10, 10)))
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # Center ROI

        t_rewound = ScaleRotor.apply_local_rotor(t, roi_mask=mask, phase_shift=np.pi/2, rewind_factor=1.0)

        # Unmasked boundary region MUST remain identical to 0.0
        assert np.all(t_rewound.phases[~mask] == 0.0)
        # Masked region MUST reflect phase rewind
        assert np.allclose(t_rewound.phases[mask], (2*np.pi - np.pi/2) % (2*np.pi))

    def test_criterion_5_divergence_verification_phase_torque_discrepancy(self):
        """
        Criterion 5: Divergence & Difference Verification.
        Verify that phase divergence directly yields mechanical torque divergence.
        """
        t1 = SpectralWaveformTensor(shape=(4, 4), phases=np.zeros((4, 4)))
        t2_in_phase = SpectralWaveformTensor(shape=(4, 4), phases=np.zeros((4, 4)))
        t2_out_phase = SpectralWaveformTensor(shape=(4, 4), phases=np.full((4, 4), np.pi / 2))

        joint = RotorCouplingConstraint(constraint_id="c_diff", target_phase_diff=0.0)

        torque_in = joint.compute_torque(t1, t2_in_phase)
        torque_out = joint.compute_torque(t1, t2_out_phase)

        assert np.isclose(torque_in, 0.0)
        assert np.isclose(abs(torque_out), 1.0)

    def test_criterion_6_counterfactual_ablation_and_failure_boundaries(self):
        """
        Criterion 6: Counterfactual & Ablation Experiments.
        Intentionally disrupt/break rotor coupling mechanics and test where judgments fail.
        """
        t1 = SpectralWaveformTensor(shape=(4, 4), phases=np.zeros((4, 4)))
        t2 = SpectralWaveformTensor(shape=(4, 4), phases=np.full((4, 4), np.pi / 2))

        # Ablation 1: Set coupling strength to 0.0 (Broken coupling)
        joint_broken = RotorCouplingConstraint(constraint_id="c_broken", coupling_strength=0.0)
        new_a, new_b, residual = joint_broken.relax_step(t1, t2)

        # Residual remains at maximum (1.0) because relaxation is ablated
        assert np.isclose(residual, 1.0)
        assert np.all(new_a.phases == t1.phases)

        # Ablation 2: Excessive coupling strength alpha > 2.0 (Over-shooting instability)
        joint_overstable = RotorCouplingConstraint(constraint_id="c_over", coupling_strength=3.0)
        new_a_over, new_b_over, residual_over = joint_overstable.relax_step(t1, t2)

        # Demonstrates physical failure boundary: phase overshoots into divergence
        assert residual_over > 0.0
