"""
test_emergent_narrative_ecosystem.py
======================================
Tests for EmergentNarrativeEcosystem module in modules/causal_game_engine/emergent_narrative_ecosystem.py.
"""

import pytest
import numpy as np
from modules.causal_game_engine.emergent_narrative_ecosystem import (
    ConceptVectorSpace,
    SelfGraph,
    EpisodicMemory,
    ZeitgeistField,
    DynamicTelosEngine,
    GenerativeVoiceFilter,
    MacroEmergenceObserver
)


def test_concept_vector_space():
    space = ConceptVectorSpace(dimension=32, seed=123)
    vec_free = space.get_vector("자유")
    assert isinstance(vec_free, np.ndarray)
    assert len(vec_free) == 32
    assert pytest.approx(np.linalg.norm(vec_free), abs=1e-5) == 1.0

    # Dynamic concept registration without pre-defined axis
    vec_custom = space.register_concept("고대의 유물")
    assert len(vec_custom) == 32
    assert "고대의 유물" in space.concepts

    # Composite vector calculation
    comp = space.compute_composite_vector({"자유": 0.8, "질서": 0.2})
    assert len(comp) == 32
    assert pytest.approx(np.linalg.norm(comp), abs=1e-5) == 1.0


def test_self_graph_and_experience_crystallization():
    space = ConceptVectorSpace(dimension=32)
    npc = SelfGraph(
        npc_id="npc_01",
        name="테스트 농민",
        archetype_title="굶주린 농민",
        vector_space=space,
        innate_traits={"altruism": 0.2, "cynicism": 0.8, "ambition": 0.1},
        initial_concepts={"생존": 0.6},
        stubbornness=0.3
    )

    # Experience crystallization under famine event
    memory = npc.crystallize_experience(
        event_id="evt_01",
        description="마을 가뭄과 기근 발생",
        event_concepts={"기근": 0.9, "배고픔": 0.8},
        environmental_pressure_name="소빙하기 기근",
        timestamp=1.0
    )

    assert isinstance(memory, EpisodicMemory)
    assert len(npc.episodic_memories) == 1
    # Cynical NPC should crystallize self-preservation / betrayal / egoism
    assert "이기" in npc.concept_weights or "생존" in npc.concept_weights
    assert npc.concept_weights["생존"] >= 0.6


def test_soul_snapshot_serialization():
    space = ConceptVectorSpace(dimension=32)
    npc = SelfGraph(
        npc_id="npc_hero",
        name="늙은 학자 헤르만",
        archetype_title="은둔 학자",
        vector_space=space,
        innate_traits={"altruism": 0.9, "cynicism": 0.1, "ambition": 0.2},
        initial_concepts={"학식": 0.8, "질서": 0.5},
        stubbornness=0.7
    )

    npc.crystallize_experience(
        event_id="evt_scholar",
        description="고서적 발견",
        event_concepts={"학식": 0.9},
        environmental_pressure_name="문화 부흥",
        timestamp=10.0
    )

    snapshot = npc.serialize_soul_snapshot()
    assert snapshot["npc_id"] == "npc_hero"
    assert snapshot["name"] == "늙은 학자 헤르만"
    assert "학식" in snapshot["concept_weights"]
    assert len(snapshot["episodic_memories"]) == 1

    # Deserialize back into a restored SelfGraph
    restored_npc = SelfGraph.deserialize_soul_snapshot(snapshot, space)
    assert restored_npc.npc_id == npc.npc_id
    assert restored_npc.name == npc.name
    assert restored_npc.concept_weights == npc.concept_weights
    assert len(restored_npc.episodic_memories) == 1
    assert restored_npc.episodic_memories[0].event_id == "evt_scholar"


def test_zeitgeist_field_and_dynamic_telos_engine():
    space = ConceptVectorSpace(dimension=32)
    zeitgeist = ZeitgeistField(
        field_id="zg_famine",
        name="소빙하기 기근",
        intensity=0.8,
        concept_signature={"기근": 0.9, "생존": 0.8}
    )

    telos_engine = DynamicTelosEngine(space)

    # NPC with high survival drive
    npc = SelfGraph(
        npc_id="npc_mercenary",
        name="거친 용병 잭",
        archetype_title="거친 용병",
        vector_space=space,
        innate_traits={"altruism": 0.1, "cynicism": 0.9, "ambition": 0.4},
        initial_concepts={"생존": 0.9, "이기": 0.7},
        stubbornness=0.2
    )
    npc.physical_needs["hunger"] = 90.0  # Very hungry

    candidates = [
        {
            "action_id": "act_pray",
            "title": "구원 기복 기도",
            "concepts": {"신앙": 0.9, "희생": 0.8},
            "hunger_relief": 0.0
        },
        {
            "action_id": "act_black_market",
            "title": "지하 암시장 강탈 및 곡물 확보",
            "concepts": {"생존": 0.9, "이기": 0.8},
            "hunger_relief": 0.9
        }
    ]

    result = telos_engine.evaluate_and_choose_action(npc, zeitgeist, candidates)
    chosen = result["chosen_action"]
    assert chosen["action_id"] == "act_black_market"
    assert npc.physical_needs["hunger"] < 90.0


def test_generative_voice_filter():
    space = ConceptVectorSpace(dimension=32)
    npc = SelfGraph(
        npc_id="npc_mercenary",
        name="용병 잭",
        archetype_title="거친 용병",
        vector_space=space,
        initial_concepts={"생존": 0.9}
    )

    action = {"action_id": "act_black_market", "title": "암시장 열기", "concepts": {"생존": 0.9}}
    dialogue = GenerativeVoiceFilter.render_dialogue(npc, action, "소빙하기 기근")
    assert isinstance(dialogue, str)
    assert len(dialogue) > 0


def test_macro_emergence_observer():
    space = ConceptVectorSpace(dimension=32)
    zeitgeist = ZeitgeistField(
        field_id="zg_war",
        name="전란의 시대",
        intensity=0.9,
        concept_signature={"전쟁": 0.9, "혁명": 0.8}
    )

    observer = MacroEmergenceObserver(space)

    action_results = [
        {
            "npc_name": "농민 A",
            "chosen_action": {"title": "영주 창고 선동 약탈", "concepts": {"혁명": 0.9, "생존": 0.7}}
        },
        {
            "npc_name": "농민 B",
            "chosen_action": {"title": "영주 창고 선동 약탈", "concepts": {"혁명": 0.9, "생존": 0.7}}
        },
        {
            "npc_name": "용병 C",
            "chosen_action": {"title": "암시장 기회주의 수탈", "concepts": {"이기": 0.8, "생존": 0.9}}
        }
    ]

    macro_record = observer.observe_turn(turn_number=1, zeitgeist_field=zeitgeist, npc_action_results=action_results)
    assert macro_record["turn"] == 1
    assert macro_record["total_micro_agents"] == 3
    assert "혁명" in macro_record["emergent_wave_title"] or "민중" in macro_record["emergent_wave_title"]
    assert len(observer.history_records) == 1
