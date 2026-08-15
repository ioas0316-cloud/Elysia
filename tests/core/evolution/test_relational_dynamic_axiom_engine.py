import pytest
from core.evolution.relational_dynamic_axiom_engine import (
    RelationalDynamicAxiomEngine,
    EmbodiedVirtualEnvironment
)

def test_embodied_virtual_environment():
    env = EmbodiedVirtualEnvironment(mass=1.0, stiffness=10.0, damping=0.5)
    state0 = env.step(external_force=1.0)
    assert "position" in state0
    assert "velocity" in state0
    assert "energy" in state0

    # Test do-intervention
    env.do_intervention("stiffness", 50.0)
    assert env.stiffness == 50.0

def test_locality_constraint_unlocking():
    engine = RelationalDynamicAxiomEngine(relativization_threshold=0.5)

    # Initial state: hooke_law and stiffness are Axiom Axes
    assert engine.nodes["stiffness"].is_axis is True
    assert engine.nodes["mass"].is_axis is True

    # High tension observation with intervention_node="stiffness"
    env_state = {"position": 5.0, "velocity": 2.0, "time": 0.1}
    prediction = {"position": 0.0, "velocity": 0.0} # Tension = 7.0 > 0.5

    trace = engine.process_observation(env_state, prediction, intervention_node="stiffness")

    # LOCALITY CONSTRAINT: stiffness and hooke_law (connected edge) should unlock, but mass should remain locked!
    assert "stiffness" in trace["unlocked_nodes"] or "hooke_law" in trace["unlocked_nodes"]
    assert engine.nodes["stiffness"].is_axis is False
    assert engine.nodes["mass"].is_axis is True  # Unaaffected axiom stays locked as Anchor!

def test_least_action_recrystallization():
    engine = RelationalDynamicAxiomEngine(condensation_threshold=0.8)

    # Manually unlock a node into variable x
    engine.nodes["stiffness"].is_axis = False
    engine.nodes["stiffness"].invariance_score = 0.75
    engine.nodes["stiffness"].resistor_x = 0.5

    # Low error steps to trigger re-crystallization
    env_state = {"position": 0.0, "velocity": 0.0, "time": 0.1}
    prediction = {"position": 0.0, "velocity": 0.0}

    for _ in range(5):
        trace = engine.process_observation(env_state, prediction)

    # Should have re-crystallized into an Axis!
    assert engine.nodes["stiffness"].is_axis is True
    assert engine.nodes["stiffness"].resistor_x == engine.min_resistor_x

def test_backtrace_projection():
    engine = RelationalDynamicAxiomEngine()
    proj = engine.backtrace_projection()
    assert "RELATIONAL CAUSAL PROJECTION" in proj
    assert "Locked Relational Axiom Axes" in proj
