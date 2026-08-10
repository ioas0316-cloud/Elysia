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

def test_causal_director_orchestrator():
    from core.physics.causal_mmorpg_sandbox import CausalDirectorOrchestrator

    orchestrator = CausalDirectorOrchestrator()

    # Simulate sandbox reports under different tension / velocity / chromatic states
    report_low = {
        "max_tension_gap": 0.0,
        "mean_resonance": 1.0,
        "refraction_index": 0.0,
        "mean_velocity_norm": 0.0,
        "chromatics": [[0.33, 0.33, 0.34]]
    }

    script_low = orchestrator.orchestrate(report_low)

    # 1. Low tension checks
    assert script_low["camera"]["shake_intensity"] == 0.0
    assert script_low["vfx"]["shader_distortion"] == 0.0
    assert script_low["camera"]["field_of_view"] == 60.0 # base fov with zero velocity
    assert script_low["vfx"]["particle_emission_rate"] == 1.0 # baseline emission rate

    report_high = {
        "max_tension_gap": 0.8,
        "mean_resonance": 0.2,
        "refraction_index": 0.0,
        "mean_velocity_norm": 5.0,
        "chromatics": [[0.8, 0.1, 0.1]] # High red flux
    }

    script_high = orchestrator.orchestrate(report_high)

    # 2. High tension / velocity checks
    assert script_high["camera"]["shake_intensity"] > 0.0
    assert script_high["vfx"]["shader_distortion"] == pytest.approx(0.8, rel=1e-5)
    assert script_high["camera"]["field_of_view"] < 60.0 # field of view zooms in (smaller angle)
    assert script_high["vfx"]["particle_emission_rate"] > 1.0 # particle emission increases
    assert script_high["camera"]["color_tint"][0] > script_high["camera"]["color_tint"][1] # red tint is dominant
