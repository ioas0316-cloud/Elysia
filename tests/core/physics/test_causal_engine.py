import numpy as np
import pytest
from core.physics.causal_engine import (
    CausalNode,
    TransitionRule,
    CausalState,
    StateDelta,
    CausalEngine
)

def test_causal_engine_initialization():
    """Verify that the engine properly configures nodes and adjacency relations."""
    nodes = [
        CausalNode(id="A", capacity=2.0, chromatic_base=(0.5, 0.5, 0.0)),
        CausalNode(id="B", capacity=1.0, chromatic_base=(0.0, 1.0, 0.0))
    ]
    rules = [
        TransitionRule(source_id="A", target_id="B", max_flow_rate=0.4, conductance=1.5)
    ]

    engine = CausalEngine(nodes, rules)

    assert "A" in engine.nodes
    assert "B" in engine.nodes
    assert len(engine.adjacency["A"]) == 1
    assert "B" in engine.adjacency["A"]

    # Confirm initial state potentials are 0.0
    assert engine.state.potentials["A"] == 0.0
    assert engine.state.potentials["B"] == 0.0

    # Check chromatic values normalized properly
    assert np.allclose(engine.state.chromatics["A"], [0.5, 0.5, 0.0])


def test_causal_engine_action_rectification_and_integration():
    """Verify that the engine rectifies invalid agent proposals and integrates valid ones."""
    nodes = [
        CausalNode(id="A", capacity=1.0),
        CausalNode(id="B", capacity=1.5)
    ]
    rules = [
        TransitionRule(source_id="A", target_id="B", max_flow_rate=0.5)
    ]
    engine = CausalEngine(nodes, rules)

    # Agent 1 (Stateless, transient): tries to set potential past capacity
    def agent_action_1(state: CausalState) -> StateDelta:
        return StateDelta(
            potential_diffs={"A": 2.5, "B": 0.5}, # A is capped at 1.0, B is OK (0.5 < 1.5)
            velocity_diffs={"A": {"B": 0.1}},
            chromatic_diffs={"A": np.array([0.1, 0.0, -0.1], dtype=np.float32)}
        )

    rectified_1 = engine.apply_action(agent_action_1)

    # Rectified potential delta for A should only be 1.0 (from 0.0 initial)
    assert rectified_1.potential_diffs["A"] == 1.0
    assert rectified_1.potential_diffs["B"] == 0.5

    assert engine.state.potentials["A"] == 1.0
    assert engine.state.potentials["B"] == 0.5
    assert engine.state.velocities["A"]["B"] == 0.1

    # Agent 2 (Stateless, transient): tries to trigger flow along a non-existent path
    def agent_action_2(state: CausalState) -> StateDelta:
        return StateDelta(
            potential_diffs={"A": -0.5},
            velocity_diffs={"B": {"A": 0.2}, "A": {"C_invalid": 0.5}} # B->A flow is OK, A->C is invalid
        )

    rectified_2 = engine.apply_action(agent_action_2)

    assert "C_invalid" not in rectified_2.velocity_diffs.get("A", {})
    assert rectified_2.potential_diffs["A"] == -0.5
    assert engine.state.potentials["A"] == 0.5


def test_causal_engine_continuous_step_flow():
    """Verify that the continuous physical flow correctly relaxes the potentials over time."""
    # Place a potential difference between A and B
    nodes = [
        CausalNode(id="A", capacity=1.0),
        CausalNode(id="B", capacity=1.0)
    ]
    rules = [
        TransitionRule(source_id="A", target_id="B", max_flow_rate=0.2, conductance=1.0)
    ]
    engine = CausalEngine(nodes, rules)

    # Set potential on A to 1.0, B to 0.0
    engine.state.potentials["A"] = 1.0
    engine.state.potentials["B"] = 0.0

    # Run continuous step: potential difference drives flow velocity, transferring potential
    # First step: velocity starts from 0, accelerates
    engine.step(dt=0.5)

    # Flow should happen from A to B
    assert engine.state.velocities["A"]["B"] > 0.0
    assert engine.state.potentials["A"] < 1.0
    assert engine.state.potentials["B"] > 0.0

    # The sum of potentials should be conserved
    total_potential = engine.state.potentials["A"] + engine.state.potentials["B"]
    assert pytest.approx(total_potential) == 1.0

    # Let the system run to equilibrium
    for _ in range(20):
        engine.step(dt=0.1)

    # Over time, potentials should equalize towards 0.5 each
    assert abs(engine.state.potentials["A"] - 0.5) < 0.1
    assert abs(engine.state.potentials["B"] - 0.5) < 0.1
