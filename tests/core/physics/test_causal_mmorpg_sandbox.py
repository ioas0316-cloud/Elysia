import pytest
import numpy as np
from core.physics.causal_mmorpg_sandbox import CausalSandboxAgent, ContinuousWorldManifold, BranchlessResonanceScheduler

def test_causal_sandbox_agent_rotor_rotation():
    agent = CausalSandboxAgent("agent_1", "TestNPC")

    # Peaceful state has highest value in rotor[0]
    state, score = agent.get_action_state()
    assert state == "PEACEFUL"
    assert score == 1.0

    # Rotate rotor along Y axis (index 2 corresponds to FLEE/FEAR if oriented that way)
    axis = np.array([0.0, 1.0, 0.0])
    # 90 degrees rotation (pi/2)
    agent.rotate_rotor(np.pi / 2, axis)

    state, score = agent.get_action_state()
    # Rotation changes values and should lead to another action state or distinct component
    assert np.linalg.norm(agent.rotor) == pytest.approx(1.0, rel=1e-5)

def test_continuous_world_manifold():
    manifold = ContinuousWorldManifold(size=100.0, sigma=10.0)
    manifold.inject_potential(np.array([10.0, 10.0, 0.0]), 5.0)

    # Exact potential at node position should be max (5.0)
    pot_at_node = manifold.get_potential_at(np.array([10.0, 10.0, 0.0]))
    assert pot_at_node == pytest.approx(5.0, rel=1e-5)

    # Gradient at the center of node should be near zero (symmetric maxima)
    grad_at_node = manifold.get_gradient_at(np.array([10.0, 10.0, 0.0]))
    assert np.all(np.abs(grad_at_node) < 1e-5)

def test_branchless_resonance_scheduler_relaxation():
    manifold = ContinuousWorldManifold(size=100.0, sigma=15.0)
    scheduler = BranchlessResonanceScheduler(manifold)

    player = CausalSandboxAgent("player", "Player", is_player=True, position=np.array([0.0, 0.0, 0.0]))
    npc = CausalSandboxAgent("npc", "NPC", is_player=False, position=np.array([5.0, 0.0, 0.0]))

    scheduler.add_agent(player)
    scheduler.add_agent(npc)

    # Step physics simulation without branchings
    report = scheduler.step(dt=0.1)

    assert report["active_agents"] == 2
    assert "mean_resonance" in report
    assert "max_tension_gap" in report
