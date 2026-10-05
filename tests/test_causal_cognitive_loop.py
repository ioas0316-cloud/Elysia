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
    assert len(agent.cognitive_history) == 1


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
    # Verify agent reacted to war shift
    agent_status = snapshot_war["cognitive_agents"][0]
    assert agent_status["dissonance"] > 0.0


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
