import pytest
import math
from core.engine.meta_causal_map import (
    DifferentialBoundMechanism,
    HarmonicConservationMechanism,
    SymbolicIntuitionMechanism,
    MetaCausalEngine,
    MechanismNode,
)


def test_differential_bound_mechanism_relaxation():
    bound = DifferentialBoundMechanism("bound_test", max_diff=2.0)
    bound.state[0] = 10.0
    bound.state[1] = 0.0

    initial_residual = bound.evaluate_residual()
    assert initial_residual > 0.0

    for _ in range(50):
        bound.project_structural_relaxation(learning_rate=1.0)
        bound.apply_deltas()

    final_residual = bound.evaluate_residual()
    diff = abs(bound.state[0] - bound.state[1])
    assert final_residual < 1e-3
    assert diff <= 2.01


def test_harmonic_conservation_mechanism_relaxation():
    harmonic = HarmonicConservationMechanism("harmonic_test", target_sum=15.0)
    harmonic.state[0] = 10.0
    harmonic.state[1] = 8.0
    harmonic.state[2] = 6.0  # Sum = 24.0, target = 15.0

    initial_residual = harmonic.evaluate_residual()
    assert initial_residual > 0.0

    for _ in range(20):
        harmonic.project_structural_relaxation(learning_rate=1.0)
        harmonic.apply_deltas()

    final_residual = harmonic.evaluate_residual()
    assert final_residual < 1e-3
    assert abs(sum(harmonic.state) - 15.0) < 1e-2


def test_meta_causal_engine_ecosystem():
    engine = MetaCausalEngine()

    bound = DifferentialBoundMechanism("mech_bound", max_diff=1.0)
    bound.state[0] = 5.0
    bound.state[1] = 1.0  # Diff = 4.0

    harmonic = HarmonicConservationMechanism("mech_harmonic", target_sum=10.0)
    harmonic.state[0] = 4.0
    harmonic.state[1] = 4.0
    harmonic.state[2] = 4.0  # Sum = 12.0

    engine.add_mechanism(bound)
    engine.add_mechanism(harmonic)
    engine.add_binding("mech_bound", "mech_harmonic", coupling_weight=0.1)

    initial_residual = engine.compute_total_residual()
    assert initial_residual > 0.0

    iterations = engine.step_convergence(max_iterations=100, tolerance=1e-3, learning_rate=0.5)
    final_residual = engine.compute_total_residual()

    assert iterations < 100
    assert final_residual < initial_residual

    contributions = engine.introspect_causal_contributions()
    assert "mech_bound" in contributions
    assert "mech_harmonic" in contributions


def test_symbolic_intuition_mechanism_cancellation():
    intuition = SymbolicIntuitionMechanism("concept_grounding", concept_a="Word_X", concept_b="Word_Y")
    intuition.state[0] = 7.0
    intuition.state[1] = -2.0  # Sum = 5.0 (Violation from equilibrium x + y = 0)

    initial_residual = intuition.evaluate_residual()
    assert initial_residual > 0.0

    for _ in range(20):
        intuition.project_structural_relaxation(learning_rate=1.0)
        intuition.apply_deltas()

    final_residual = intuition.evaluate_residual()
    assert final_residual < 1e-4
    assert abs(intuition.state[0] + intuition.state[1]) < 1e-3


def test_if_else_vs_constraint_relaxation():
    """
    Compares traditional if-else branching against constraint relaxation.
    In if-else, abrupt clipping causes step discontinuity.
    In constraint relaxation, equilibrium smoothly absorbs residual error.
    """
    # 1. Traditional if-else
    def traditional_if_else_control(value: float, max_limit: float) -> float:
        if value > max_limit:
            return max_limit
        return value

    clipped_val = traditional_if_else_control(10.0, 5.0)
    assert clipped_val == 5.0

    # 2. Constraint Relaxation
    bound = DifferentialBoundMechanism("smooth_bound", max_diff=5.0)
    bound.state[0] = 10.0
    bound.state[1] = 0.0  # Diff = 10.0, max = 5.0

    for _ in range(50):
        bound.project_structural_relaxation(learning_rate=1.0)
        bound.apply_deltas()

    diff = abs(bound.state[0] - bound.state[1])
    assert abs(diff - 5.0) < 0.05
    # Both states shifted smoothly to accommodate constraint
    assert bound.state[0] < 10.0
    assert bound.state[1] > 0.0
