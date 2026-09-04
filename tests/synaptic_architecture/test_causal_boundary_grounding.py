"""
Unit tests for Causal Boundary Tensor, Causal DNA Connectors, 0th-Order Archetypal Cognition Engine,
Minimal Causal Engine, Differential Perceptual Engine, and Causal Grounding Pipeline.
"""

import pytest
import torch
import numpy as np

from synaptic_architecture import (
    CausalBoundaryTensor,
    MinimalCausalEngine,
    ArchetypalCognitionEngine,
    SensoryInvariantModeling,
    DifferentialPerceptualEngine,
    CausalGroundingPipeline
)


class TestCausalBoundaryTensor:
    """Tests for CausalBoundaryTensor arithmetic DNA connectors and dynamic logic gates."""

    def test_arithmetic_dna_connectors(self):
        t1 = CausalBoundaryTensor(state=[1.0, 2.0], boundary_phase=[0.0, 0.0], value_ground=1.0)
        t2 = CausalBoundaryTensor(state=[0.5, 1.0], boundary_phase=[0.0, 0.0], value_ground=1.0)

        # Addition: Binding & Synthesis with phase resonance
        res_add = t1 + t2
        assert torch.allclose(res_add.state, torch.tensor([1.5, 3.0]))

        # Subtraction: Cleavage & Gradient Formation
        res_sub = t1 - t2
        assert torch.allclose(res_sub.state, torch.tensor([0.5, 1.0]))

        # Multiplication: Replication & Scaling
        res_mul = t1 * t2
        assert torch.allclose(res_mul.state, torch.tensor([0.5, 2.0]))

        # Division: Differentiation & Seed Formation
        res_div = t1 / t2
        assert torch.allclose(res_div.state, torch.tensor([2.0, 2.0]))

    def test_dynamic_logic_gates(self):
        t1 = CausalBoundaryTensor(state=[1.0, 0.0], boundary_phase=[0.0, 0.0])
        t2 = CausalBoundaryTensor(state=[1.0, 1.0], boundary_phase=[0.0, np.pi])

        # AND Gate: Strict Resonance & Inter-fidelity
        res_and = t1.and_gate(t2)
        assert isinstance(res_and, CausalBoundaryTensor)

        # OR Gate: Boundary Permeability & Acceptance
        res_or = t1.or_gate(t2)
        assert isinstance(res_or, CausalBoundaryTensor)

        # NOT Gate: Inversion & Polarity Shift
        res_not = t1.not_gate()
        assert torch.allclose(res_not.state, torch.tensor([0.0, 1.0]))

        # XOR Gate: Differential Sensing & Motion Triggering
        res_xor = t1.xor_gate(t2)
        assert res_xor.state.shape == torch.Size([2])


class TestMinimalCausalEngine:
    """Tests for Minimal Causal Engine Intention -> Deformation -> Friction -> Self-Correction loop."""

    def test_minimal_causal_engine_step(self):
        engine = MinimalCausalEngine(state_dim=3, constraint_limit=1.5)
        intent = torch.tensor([1.0, -1.0, 0.5])

        res = engine.step(intent)
        assert "S_next" in res
        assert "residual_friction" in res
        assert "R_actual" in res
        assert res["S_current"].shape == torch.Size([3])


class TestArchetypalCognitionEngine:
    """Tests for 0th-Order Archetypal Cognition Engine."""

    def test_archetypal_principles(self):
        engine = ArchetypalCognitionEngine(value_ground=1.0)

        # Identity & Difference
        s1 = torch.tensor([1.0, 2.0, 3.0])
        s2 = torch.tensor([1.0, 2.0, 3.01])
        id_res = engine.observe_identity_and_difference(s1, s2, invariance_threshold=0.05)
        assert id_res["is_identity"] is True

        # Connectivity (delta_A -> delta_B)
        dA = torch.tensor([1.0, 0.0])
        dB = torch.tensor([0.9, 0.1])
        conn_res = engine.trace_connectivity(dA, dB)
        assert conn_res["is_connected"] is True

        # Relationality (Mutual constraint on DOF)
        rel_res = engine.evaluate_relationality(torch.tensor([1.5]), torch.tensor([0.5]))
        assert rel_res["has_relationship"] is True

        # Recursive Cognition Loop
        cog_res = engine.recursive_cognition_loop(torch.tensor([0.1, 0.2, -0.1, 0.05]))
        assert "updated_internal_state" in cog_res


class TestSensoryAndDifferentialPerception:
    """Tests for Sensory Invariant Modeling & Differential Perceptual Engine."""

    def test_topological_error_detection(self):
        diff_engine = DifferentialPerceptualEngine()

        # Normal hand (5 digits)
        normal_hand = {"num_digits": 5, "joint_angles": torch.tensor([0.1]*5)}
        res_normal = diff_engine.compare_and_discern(normal_hand)
        assert res_normal["has_topological_error"] is False

        # Anomalous hand (6th finger error)
        anomalous_hand = {"num_digits": 6, "joint_angles": torch.tensor([0.1]*6)}
        res_anomalous = diff_engine.compare_and_discern(anomalous_hand)
        assert res_anomalous["has_topological_error"] is True
        assert "6th finger" in res_anomalous["error_type"][0]


class TestCausalGroundingPipeline:
    """Tests for 4-Stage Causal Grounding Pipeline."""

    def test_pipeline_normal_and_veto(self):
        pipeline = CausalGroundingPipeline(veto_threshold=2.5, value_ground=1.0)

        # Normal aligned input signal
        normal_input = CausalBoundaryTensor(
            state=torch.ones(4),
            boundary_phase=torch.zeros(4)
        )
        res_normal = pipeline.process(normal_input)
        assert res_normal["status"] == "CAUSAL_RESONANCE_ESTABLISHED"
        assert res_normal["output"] is not None

        # Destructive input with 6th finger topological anomaly -> Veto execution
        destructive_input = CausalBoundaryTensor(
            state=torch.tensor([10.0, -10.0, 5.0, -5.0]),
            boundary_phase=torch.tensor([np.pi, np.pi, np.pi, np.pi])
        )
        anomalous_meta = {"num_digits": 6}
        res_veto = pipeline.process(destructive_input, metadata=anomalous_meta)
        assert res_veto["status"] == "VETO_EXECUTED"
        assert res_veto["output"] is None
