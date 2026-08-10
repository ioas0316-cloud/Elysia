import pytest
import numpy as np
from core.physics.causal_mmorpg_sandbox import (
    CausalSandboxAgent,
    ContinuousWorldManifold,
    BranchlessResonanceScheduler,
    CausalDirectorOrchestrator
)

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

def test_metaphorical_refraction_dimensional_phase_inversion():
    manifold = ContinuousWorldManifold(size=100.0, sigma=15.0)
    scheduler = BranchlessResonanceScheduler(manifold)

    player = CausalSandboxAgent("player", "Player", is_player=True, position=np.array([0.0, 0.0, 0.0]))
    npc = CausalSandboxAgent("npc", "NPC", is_player=False, position=np.array([5.0, 0.0, 0.0]))

    scheduler.add_agent(player)
    scheduler.add_agent(npc)

    # Ingress of literal word (refraction should be 0.0)
    report_literal = scheduler.step(dt=0.1, input_concept="apple")
    assert report_literal["refraction_index"] == 0.0

    # Ingress of abstract/metaphorical word (refraction should be 1.0)
    # The forces will be projected to mental_positions instead of positions.
    report_abstract = scheduler.step(dt=0.1, input_concept="grace of love")
    assert report_abstract["refraction_index"] == 1.0

    # Verify mental positions contain values
    assert "mental_positions" in report_abstract


def test_causal_director_orchestration():
    orchestrator = CausalDirectorOrchestrator(base_fov=60.0)

    dummy_report = {
        "max_tension_gap": 0.5,
        "mean_resonance": 0.8,
        "refraction_index": 0.0,
        "chromatics": [[0.5, 0.2, 0.3], [0.4, 0.3, 0.3]]
    }

    script = orchestrator.orchestrate(dummy_report)

    # Validate Camera FOV, Shake, and low pass cutoff
    assert "camera" in script
    assert "vfx" in script
    assert "audio" in script

    assert script["camera"]["field_of_view"] < 60.0 # Zoom-in due to high resonance
    assert script["vfx"]["particle_emission_rate"] > 1.0
    assert script["audio"]["low_pass_cutoff"] < 20000.0
