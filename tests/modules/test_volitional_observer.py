"""
Unit Tests: Volitional Observer & Digital Twin Environment
==========================================================
의지적 어텐션, 도구 신체화, 관문 잠금 해제 및 체화된 지식 축적 단위 테스트.
"""

import pytest
from core.topology.causal_stem_branch_engine import CausalNode, NodeType
from modules.causal_game_engine.volitional_observer import (
    VolitionalObserver,
    CausalAttentionEngine,
    INVARIANT_TOOL_KEY
)
from modules.causal_game_engine.digital_twin_world import DigitalTwinWorld


def test_attention_engine_lens_shift():
    attn = CausalAttentionEngine()

    # 상황 1: 문이 잠겨있고 열쇠가 없을 때 -> 도구로 어텐션 집중
    choice1 = attn.resolve_attention(
        current_pos=(2, 2),
        goal_pos=(18, 5),
        is_blocked_by_gate=True,
        visible_tools=[{"pos": (5, 8)}],
        has_tool=False
    )
    assert choice1 == "ATTEND_TO_TOOL"

    # 상황 2: 열쇠를 신체화했을 때 -> 목적지 성소로 어텐션 복귀
    choice2 = attn.resolve_attention(
        current_pos=(5, 8),
        goal_pos=(18, 5),
        is_blocked_by_gate=False,
        visible_tools=[],
        has_tool=True
    )
    assert choice2 == "ATTEND_TO_GOAL"


def test_tool_assimilation_expands_causal_graph():
    observer = VolitionalObserver("ego_tester", x=2, y=2, goal_x=18, goal_y=5)
    orig_node_count = len(observer.graph.nodes)
    orig_edge_count = len(observer.graph.edges)

    key_node = CausalNode(
        node_id="key_card_1",
        node_type=NodeType.STEM,
        invariant_signature=INVARIANT_TOOL_KEY,
        payload={"name": "Golden_Key"}
    )
    observer.assimilate_tool("key_card_1", key_node)

    assert "key_card_1" in observer.assimilated_tools
    assert len(observer.graph.nodes) == orig_node_count + 1
    assert len(observer.graph.edges) == orig_edge_count + 1
    assert observer.current_lens == "Constraint_Unlocker"


def test_digital_twin_gate_unlocking_and_mission_completion():
    world = DigitalTwinWorld(width=22, height=12)
    observer = VolitionalObserver("hero_ego", x=2, y=2, goal_x=18, goal_y=5)
    world.spawn_observer(observer)

    # 40 사이클 실행
    for step in range(40):
        res = world.step()
        if observer.mission_completed:
            break

    assert observer.mission_completed is True
    assert world.gate_unlocked is True
    assert "key_card" in observer.assimilated_tools
    assert observer.x == 18 and observer.y == 5
    assert len(observer.epistemic_trajectory) > 0
