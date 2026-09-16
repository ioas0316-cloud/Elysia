"""
Unit Tests: Causal Automaton & Sandbox
======================================
기계적 맞물림과 더듬이 반사를 통한 자율 이동/회피 단위 테스트.
"""

import pytest
from modules.causal_game_engine.causal_automaton import (
    CausalAutomaton,
    INVARIANT_POWER_CORE,
    INVARIANT_SENSOR_WHISKER
)
from modules.causal_game_engine.automaton_sandbox import AutomatonSandbox


def test_causal_automaton_assembly():
    bot = CausalAutomaton(automaton_id="test_bot", x=5, y=5, dir_idx=1)
    # 5개 핵심 부품 노드 확인
    assert len(bot.graph.nodes) == 5
    assert len(bot.graph.edges) == 3
    assert bot.graph.nodes["test_bot_power"].invariant_signature == INVARIANT_POWER_CORE
    assert bot.graph.nodes["test_bot_whisker"].invariant_signature == INVARIANT_SENSOR_WHISKER


def test_unblocked_locomotion():
    bot = CausalAutomaton(automaton_id="test_bot", x=5, y=5, dir_idx=1) # 동쪽(EAST, x+1)
    # 장애물이 없는 환경 함수
    res = bot.tick(world_obstacle_check_fn=lambda x, y: False)

    assert res["pos"] == (6, 5)
    assert res["dir"] == "EAST"
    assert not res["whisker_blocked"]
    assert "ADVANCE" in res["action"]


def test_obstacle_whisker_deflection():
    bot = CausalAutomaton(automaton_id="test_bot", x=5, y=5, dir_idx=1) # 동쪽(EAST)
    # (6, 5) 위치에 벽이 있는 환경
    def obstacle_check(x, y):
        return (x, y) == (6, 5)

    res = bot.tick(world_obstacle_check_fn=obstacle_check)

    # 위치는 침범하지 않고 그대로 (5, 5), 방향은 90도 시계방향 회전하여 SOUTH(남쪽)
    assert res["pos"] == (5, 5)
    assert res["dir"] == "SOUTH"
    assert res["whisker_blocked"]
    assert "DEFLECT" in res["action"]


def test_sandbox_boundary_enforcement():
    sandbox = AutomatonSandbox(width=10, height=10)
    bot = CausalAutomaton(automaton_id="roamer", x=8, y=5, dir_idx=1) # 동쪽 벽 (9, 5) 바로 앞
    sandbox.spawn_automaton(bot)

    # 1틱: (9, 5) 벽을 감지하고 남쪽으로 회전
    res1 = sandbox.step()
    assert res1["pos"] == (8, 5)
    assert res1["dir"] == "SOUTH"

    # 2틱: 남쪽은 열려있으므로 (8, 6)으로 전진
    res2 = sandbox.step()
    assert res2["pos"] == (8, 6)
    assert res2["dir"] == "SOUTH"
