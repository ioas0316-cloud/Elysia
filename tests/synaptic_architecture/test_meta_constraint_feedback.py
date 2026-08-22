"""
Unit tests for Meta-Constraint Feedback Loop & Causal Impedance Engine
"""

import pytest
import numpy as np
import causal_engine as ce
from synaptic_architecture.meta_constraint_feedback import MetaConstraintFeedbackLoop


def test_cpp_causal_impedance_evaluator():
    # Construct mock nodes
    n0 = ce.MacroSymbolNode()
    n0.node_id = 0
    n0.pivot_alpha = 0.0
    n0.pivot_beta = 0.0
    n0.axiom_rigidity = 0.8

    n1 = ce.MacroSymbolNode()
    n1.node_id = 1
    n1.pivot_alpha = 0.5
    n1.pivot_beta = 0.5
    n1.axiom_rigidity = 0.7

    n2 = ce.MacroSymbolNode()
    n2.node_id = 2
    n2.pivot_alpha = 1.0
    n2.pivot_beta = 0.0  # Creates sharp 90-degree turn from (0,0)->(0.5,0.5)->(1,0)
    n2.axiom_rigidity = 0.6

    nodes = [n0, n1, n2]

    # Test Curvature calculation
    straight_trajectory = [0, 1]
    bent_trajectory = [0, 1, 2]

    curv_straight = ce.CausalImpedanceEvaluator.compute_curvature(nodes, straight_trajectory)
    curv_bent = ce.CausalImpedanceEvaluator.compute_curvature(nodes, bent_trajectory)

    assert curv_straight == 0.0
    assert curv_bent > 0.0  # Bent path must yield non-zero curvature

    # Test Phase Discrepancy calculation
    target_trajectory = [0, 1, 2]
    phase_diff = ce.CausalImpedanceEvaluator.compute_phase_diff(nodes, bent_trajectory, target_trajectory)
    assert phase_diff >= 0.0

    # Full Evaluation
    impedance = ce.CausalImpedanceEvaluator.evaluate_impedance(
        nodes, bent_trajectory, target_trajectory, gamma_curvature=0.4, latency_damping=0.1, friction_threshold=0.1
    )

    assert impedance.trajectory_curvature == curv_bent
    assert impedance.latency_damped_friction > 0.0
    assert impedance.resonance_score > 0.0


def test_meta_constraint_mutator():
    mutator = ce.MetaConstraintMutator()
    rule_init = mutator.get_current_rule()

    n0 = ce.MacroSymbolNode()
    n0.node_id = 0
    n0.pivot_alpha = 0.0
    n0.pivot_beta = 0.0

    n1 = ce.MacroSymbolNode()
    n1.node_id = 1
    n1.pivot_alpha = 0.5
    n1.pivot_beta = 0.5

    n2 = ce.MacroSymbolNode()
    n2.node_id = 2
    n2.pivot_alpha = 1.0
    n2.pivot_beta = 0.0

    nodes = [n0, n1, n2]
    trajectory = [0, 1, 2]

    imp = ce.ImpedanceResult()
    imp.trajectory_curvature = 0.8
    imp.topological_phase_diff = 0.6
    imp.latency_damped_friction = 0.5
    imp.resonance_score = 0.66
    imp.requires_rule_mutation = True

    mutator.mutate_rule(imp, nodes, trajectory)

    rule_mutated = mutator.get_current_rule()

    assert mutator.get_mutation_count() == 1
    assert rule_mutated.max_reluctance_threshold < rule_init.max_reluctance_threshold
    assert rule_mutated.min_rigidity_threshold > rule_init.min_rigidity_threshold


def test_meta_constraint_feedback_loop_integration():
    engine = MetaConstraintFeedbackLoop(
        num_field_nodes=32,
        hysterons_per_dim=8,
        gamma_curvature=0.3,
        latency_damping=0.2,
        friction_threshold=0.3,
    )

    target_trajectory = [0, 2, 4]
    step_result = engine.step_meta_feedback(0, 4, target_trajectory)

    assert "best_trajectory" in step_result
    assert "trajectory_curvature" in step_result
    assert "latency_damped_friction" in step_result
    assert "resonance_score" in step_result
    assert "rule" in step_result
