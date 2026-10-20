"""
Tests for 4-Stage Causal Cognitive Loop Engine (tests/test_causal_cognitive_loop.py)
"""

import numpy as np
import pytest
from elysia_core.causal_cognitive_loop import (
    CausalCognitiveAgent,
    CausalCognitiveEngine
)
from elysia_core.unified_phase_space import (
    ResourceEntity,
    CraftedItemEntity,
    MonsterEntity
)


def test_agent_perception():
    agent = CausalCognitiveAgent("테스트 기사", np.array([0.0, -0.2, 0.8, 0.6, 0.0]))
    env_vec = np.array([0.5, 0.5, -0.5, -0.5, 0.2])

    perceived = agent.perceive_environment(env_vec)
    np.testing.assert_array_almost_equal(perceived, env_vec)
    np.testing.assert_array_almost_equal(agent.last_perceived_env, env_vec)


def test_agent_discrimination_and_phase_transition():
    # Agent with high order/origin alignment
    agent = CausalCognitiveAgent("칼하인츠", np.array([-0.5, -0.5, 0.8, 0.8, 0.0]), dissonance_threshold=0.8)

    # Mild shift -> No phase transition
    mild_env = np.array([-0.4, -0.4, 0.7, 0.7, 0.1])
    agent.perceive_environment(mild_env)
    diss, shifted = agent.judge_dissonance()

    assert not shifted
    assert diss < 0.8

    # Severe macro shift -> Triggers Quantum Phase Transition
    extreme_env = np.array([0.9, 0.8, -0.8, -0.9, 0.5])
    agent.perceive_environment(extreme_env)
    diss, shifted = agent.judge_dissonance()

    assert shifted
    assert agent.phase_transitions_count == 1
    assert agent.last_phase_shifted


def test_agent_reflection_and_adaptation():
    agent = CausalCognitiveAgent("원로 학자", np.array([0.0, 0.0, 0.5, 0.5, 0.0]), relaxation_rate=0.3)
    env_vec = np.array([0.8, 0.8, -0.5, -0.5, 0.2])

    agent.perceive_environment(env_vec)
    initial_S = np.copy(agent.S)
    updated_S = agent.reflect_and_adapt()

    # Verify vector moved closer to environmental potential
    dist_before = np.linalg.norm(initial_S - env_vec)
    dist_after = np.linalg.norm(updated_S - env_vec)
    assert dist_after < dist_before


def test_agent_reverse_projection():
    agent = CausalCognitiveAgent("혁명가 카일", np.array([0.9, 0.8, -0.7, -0.6, 0.4]), reverse_projection_weight=0.1)
    env_vec = np.array([-0.5, -0.5, 0.8, 0.8, 0.0])

    agent.perceive_environment(env_vec)
    delta_V = agent.act_and_reverse_project()

    # Delta V should point in direction of agent's phase vector minus environment
    expected_delta = 0.1 * (agent.S - env_vec)
    np.testing.assert_array_almost_equal(delta_V, expected_delta)

    # Evaluate outcome to complete the cycle and log into cognitive history & growth rings
    record = agent.evaluate_result_and_reexamine_criteria(
        prior_state=np.array([0.9, 0.8, -0.7, -0.6, 0.4]),
        actual_resulting_env=env_vec + delta_V,
        delta_V=delta_V,
        context_name="Test Reverse Projection Context"
    )
    assert len(agent.cognitive_history) == 1
    assert len(agent.growth_rings) == 1


def test_causal_cognitive_engine_full_loop():
    engine = CausalCognitiveEngine(
        env_name="신정정치 평화기",
        env_base_vec=np.array([-0.4, -0.5, 0.8, 0.8, 0.2])
    )

    agent = CausalCognitiveAgent("기사장 칼하인츠", np.array([-0.3, -0.4, 0.7, 0.8, 0.1]))
    engine.add_cognitive_agent(agent)

    ore = ResourceEntity("고위상 마나석", np.array([0.1, 0.2, 0.6, 0.4, 0.5]))
    engine.add_resource(ore)

    boss = MonsterEntity("심연의 괴수", np.array([0.5, 0.2, -0.3, -0.2, 0.1]), energy_shell=3)
    engine.add_monster(boss)

    # Initial state cycle
    snapshot = engine.step_cognitive_cycle()
    assert engine.total_cycles == 1
    assert len(snapshot["cognitive_agents"]) == 1

    # Macro Zeitgeist Shift (War & Pollution)
    war_shift = np.array([1.2, 1.0, -1.4, -1.3, 0.4])
    snapshot_war = engine.step_cognitive_cycle(macro_event_delta=war_shift)

    assert engine.total_cycles == 2
    # Verify agent reacted to war shift, re-examined criteria, and logged growth ring
    agent_status = snapshot_war["cognitive_agents"][0]
    assert agent_status["dissonance"] > 0.0
    assert agent_status["growth_rings_count"] == 2
    assert agent_status["criteria_reexaminations"] >= 1


def test_cognitive_one_page_dashboard_rendering():
    engine = CausalCognitiveEngine(
        env_name="평화와 질서의 시대",
        env_base_vec=np.array([-0.3, -0.4, 0.8, 0.8, 0.1])
    )
    agent = CausalCognitiveAgent("대사제 엘리아스", np.array([-0.2, -0.3, 0.9, 0.7, 0.4]))
    engine.add_cognitive_agent(agent)

    ore = ResourceEntity("성스러운 수정", np.array([-0.1, -0.2, 0.8, 0.5, 0.3]))
    engine.add_resource(ore)

    sword = CraftedItemEntity("성스러운 검", [ore.current_R], energy_shell_n=2)
    engine.add_crafted_item(sword)

    engine.step_cognitive_cycle()
    dashboard = engine.render_cognitive_one_page_dashboard(title_suffix="[TEST REPORT]")

    assert "CAUSAL COGNITIVE LOOP ONE-PAGE DASHBOARD" in dashboard
    assert "대사제 엘리아스" in dashboard
    assert "성스러운 수정" in dashboard
    assert "성스러운 검" in dashboard


def test_sensory_grounding_and_reach_limits():
    from elysia_core.causal_cognitive_loop import GroundedSensoryApparatus

    apparatus = GroundedSensoryApparatus(reach_distance=3.0)
    env_vec = np.array([0.6, 0.4, -0.2, 0.1, 0.5])
    agent_pos = np.array([0.0, 0.0, 0.0])

    # Within reach (dist = 1.0 <= 3.0)
    target_close = np.array([1.0, 0.0, 0.0])
    obs_close = apparatus.probe_environment(env_vec, agent_pos=agent_pos, target_pos=target_close)

    assert obs_close.is_grounded
    assert obs_close.tactile_friction > 0.4
    assert obs_close.grounding_index > 0.6

    # Beyond reach (dist = 5.0 > 3.0)
    target_far = np.array([5.0, 0.0, 0.0])
    obs_far = apparatus.probe_environment(env_vec, agent_pos=agent_pos, target_pos=target_far)

    assert not obs_far.is_grounded
    assert obs_far.tactile_friction < 0.2
    assert obs_far.grounding_index < 0.5


def test_fractal_node_3x3x3_phase_locking():
    from elysia_core.causal_cognitive_loop import FractalNode3x3x3

    grid = FractalNode3x3x3(coupling_strength=0.8)
    stimulus = np.array([0.5, 0.8, -0.2])

    initial_coherence = grid.coherence
    # Run multiple steps to allow phase locking
    for _ in range(10):
        coherence = grid.step_phase_locking(stimulus, dt=0.05)

    assert 0.0 <= coherence <= 1.0


def test_cognitive_grounding_boundary_and_hallucination_control():
    from elysia_core.causal_cognitive_loop import GroundedSensoryApparatus, CognitiveGroundingBoundary

    apparatus = GroundedSensoryApparatus(reach_distance=2.0)
    boundary = CognitiveGroundingBoundary(hallucination_threshold=0.5)

    env_vec = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    internal_belief = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sim_hypothesis = np.array([-1.0, -1.0, 1.0, 1.0, 0.5])

    # Case 1: Un-grounded (far target)
    far_pos = np.array([10.0, 0.0, 0.0])
    obs_far = apparatus.probe_environment(env_vec, agent_pos=np.array([0.0, 0.0, 0.0]), target_pos=far_pos)
    eval_far = boundary.evaluate_grounding(obs_far, internal_belief, sim_hypothesis)

    assert not eval_far["is_grounded"]
    assert eval_far["hallucination_risk"] > 0.0

    # Case 2: Grounded (close target)
    close_pos = np.array([0.5, 0.0, 0.0])
    obs_close = apparatus.probe_environment(env_vec, agent_pos=np.array([0.0, 0.0, 0.0]), target_pos=close_pos)
    eval_close = boundary.evaluate_grounding(obs_close, internal_belief, sim_hypothesis)

    assert eval_close["is_grounded"]


def test_agent_spatial_cognitive_loop():
    engine = CausalCognitiveEngine("스케일 공간 환경")
    agent = CausalCognitiveAgent("탐험가 리안", np.array([0.1, 0.2, 0.5, 0.5, 0.1]), reach_distance=3.0)
    engine.add_cognitive_agent(agent)

    positions = {
        "탐험가 리안": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    }

    snapshot = engine.step_cognitive_cycle(agent_positions=positions)
    agent_status = snapshot["cognitive_agents"][0]

    assert agent_status["grounding_index"] > 0.5
    assert "Causal Sense" in engine.render_cognitive_one_page_dashboard()
