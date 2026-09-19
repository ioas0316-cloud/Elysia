"""
demo_emergent_narrative_ecosystem.py
=====================================
CLI Demo Simulator for Elysia Emergent Narrative Ecosystem.

Simulates bottom-up micro-agency and macro-causality dynamics:
- Macro Zeitgeist Field (시대적 장): "소빙하기의 대기근 (Great Famine of the Little Ice Age)"
- Micro Agent Self-Graphs: 3 unique NPCs (학자, 용병, 농민) with distinct innate seeds & dynamic concept landscapes.
- Dynamic Telos Decision Engine: Resolves actions via inner concept resonance and physical drives.
- Generative Voice Filter: Renders decisions into distinct persona dialogue.
- Soul Snapshot Serialization: Demonstrates dynamic-to-static soul state persistence & loading.
- Macro Emergence Observer: Synthesizes bottom-up micro actions into emergent macro historical waves.
"""

import time
import json
from modules.causal_game_engine.emergent_narrative_ecosystem import (
    ConceptVectorSpace,
    SelfGraph,
    ZeitgeistField,
    DynamicTelosEngine,
    GenerativeVoiceFilter,
    MacroEmergenceObserver
)


def run_emergent_narrative_ecosystem_demo():
    print("=" * 85)
    print(" Elysia Causal Engine: Emergent Narrative Ecosystem Simulation ")
    print(" System: Zeitgeist Tensor Field × Dynamic Concept Vector Self-Graph × Macro Wave Emergence ")
    print("=" * 85)

    # 1. Initialize Core Substrates
    vector_space = ConceptVectorSpace(dimension=32, seed=2025)
    telos_engine = DynamicTelosEngine(vector_space)
    macro_observer = MacroEmergenceObserver(vector_space)

    # 2. Define Macro Zeitgeist Field
    zeitgeist_famine = ZeitgeistField(
        field_id="zg_little_ice_age",
        name="소빙하기의 대기근",
        intensity=0.85,
        concept_signature={"기근": 0.9, "생존": 0.8, "공포": 0.6}
    )

    print(f"\n[1. Zeitgeist Field Activated]")
    print(f"   - Field Name: {zeitgeist_famine.name} (Intensity: {zeitgeist_famine.intensity})")
    print(f"   - Environmental Concept Pressure: {zeitgeist_famine.concept_signature}")

    # 3. Instantiate Micro Agent Nodes with Unique Seeds
    npc_scholar = SelfGraph(
        npc_id="npc_hermann",
        name="늙은 학자 헤르만",
        archetype_title="늙은 학자",
        vector_space=vector_space,
        innate_traits={"altruism": 0.8, "cynicism": 0.1, "ambition": 0.2},
        initial_concepts={"학식": 0.8, "이성": 0.7, "질서": 0.6},
        stubbornness=0.7
    )

    npc_mercenary = SelfGraph(
        npc_id="npc_jack",
        name="거친 용병 잭",
        archetype_title="거친 용병",
        vector_space=vector_space,
        innate_traits={"altruism": 0.15, "cynicism": 0.85, "ambition": 0.4},
        initial_concepts={"생존": 0.8, "이기": 0.7},
        stubbornness=0.2
    )

    npc_peasant = SelfGraph(
        npc_id="npc_karl",
        name="분노한 농민 칼",
        archetype_title="굶주린 농민",
        vector_space=vector_space,
        innate_traits={"altruism": 0.4, "cynicism": 0.3, "ambition": 0.8},
        initial_concepts={"가족": 0.7, "생존": 0.6},
        stubbornness=0.3
    )

    agents = [npc_scholar, npc_mercenary, npc_peasant]

    print("\n[2. Micro Agent Nodes Initialized]")
    for agent in agents:
        print(f"   - [{agent.npc_id}] {agent.name} ({agent.archetype_title})")
        print(f"     Innate Traits: {agent.innate_traits}")
        print(f"     Concept Landscape: {agent.concept_weights}")

    # 4. Action Candidates Pool in Famine Situation
    action_candidates = [
        {
            "action_id": "act_pray_sacrifice",
            "title": "희생적 성당 기복 기도 및 빵 나누기",
            "concepts": {"신앙": 0.9, "이타": 0.8, "숭고": 0.8},
            "hunger_relief": 0.1
        },
        {
            "action_id": "act_black_market",
            "title": "지하 암시장 식량 수탈 및 기회주의 매점매석",
            "concepts": {"생존": 0.9, "이기": 0.85, "배신": 0.6},
            "hunger_relief": 0.95
        },
        {
            "action_id": "act_peasant_rebellion",
            "title": "영주 곡물 창고 습격 및 민중 선동 약탈",
            "concepts": {"혁명": 0.95, "야망": 0.8, "자유": 0.7},
            "hunger_relief": 0.85
        },
        {
            "action_id": "act_record_chronicle",
            "title": "기근 비망록 기록 및 이성적 원인 탐구",
            "concepts": {"학식": 0.9, "이성": 0.85, "전통": 0.6},
            "hunger_relief": 0.05
        }
    ]

    # 5. Run 5-Turn Narrative Simulation
    print("\n" + "=" * 85)
    print(" Starting 5-Turn Micro-Agency & Macro-Causality Simulation...")
    print("=" * 85)

    soul_snapshot_saved = None

    for turn in range(1, 6):
        print(f"\n>>> [TURN {turn:02d}] ----------------------------------------------------")
        turn_action_results = []

        # Increase Hunger over time
        for agent in agents:
            agent.physical_needs["hunger"] = min(100.0, agent.physical_needs["hunger"] + 18.0)

        for agent in agents:
            # 1) Crystallize Experience of turn's environmental pressure
            memory = agent.crystallize_experience(
                event_id=f"evt_turn_{turn}_{agent.npc_id}",
                description=f"Turn {turn}: {zeitgeist_famine.name} 심화 (허기 수치: {agent.physical_needs['hunger']:.1f})",
                event_concepts={"기근": 0.8, "생존": 0.7},
                environmental_pressure_name=zeitgeist_famine.name,
                timestamp=float(turn)
            )

            # 2) Dynamic Telos Decision Making
            decision = telos_engine.evaluate_and_choose_action(agent, zeitgeist_famine, action_candidates)
            chosen_act = decision["chosen_action"]

            # 3) Render Dialogue via Persona Voice Filter
            dialogue = GenerativeVoiceFilter.render_dialogue(agent, chosen_act, zeitgeist_famine.name)

            print(f"\n   👤 [{agent.name} ({agent.archetype_title})]")
            print(f"      - 허기: {agent.physical_needs['hunger']:.1f} | 주관적 해석: {memory.subjective_interpretation}")
            print(f"      - 결정된 행동: {chosen_act['title']} (Resonance Score: {decision['chosen_score']:.2f})")
            print(f"      - 페르소나 대사: {dialogue}")
            print(f"      - 갱신된 내면 가치관: {agent.concept_weights}")

            turn_action_results.append({
                "npc_name": agent.name,
                "chosen_action": chosen_act
            })

        # Save Soul Snapshot at Turn 3 for Mercerary Jack
        if turn == 3:
            soul_snapshot_saved = npc_mercenary.serialize_soul_snapshot()
            print(f"\n   💾 [SOUL SNAPSHOT SAVED] '{npc_mercenary.name}'의 동적 자아가 파일로 직렬화되었습니다.")

        # 4) Macro Emergence Observation
        macro_record = macro_observer.observe_turn(
            turn_number=turn,
            zeitgeist_field=zeitgeist_famine,
            npc_action_results=turn_action_results
        )

        print(f"\n   🌌 [상위 관측 (MACRO WAVE EMERGENCE)]")
        print(f"      - 발현된 시대적 파도: {macro_record['emergent_wave_title']}")
        print(f"      - {macro_record['macro_meaning']}")

    # 6. Restoring Soul Snapshot Demonstration
    print("\n" + "=" * 85)
    print(" Demonstration: Restoring Soul Snapshot (동적 자아 복원) ")
    print("=" * 85)
    if soul_snapshot_saved:
        restored_jack = SelfGraph.deserialize_soul_snapshot(soul_snapshot_saved, vector_space)
        print(f"   Successfully restored Soul Snapshot for [{restored_jack.name}]:")
        print(f"   - Restored Concept Weights: {restored_jack.concept_weights}")
        print(f"   - Restored Episodic Memories Count: {len(restored_jack.episodic_memories)}")
        print(f"   - Last Memory: {restored_jack.episodic_memories[-1].description}")

    print("\n" + "=" * 85)
    print(" Emergent Narrative Ecosystem CLI Simulation Completed Successfully! ")
    print("=" * 85)


if __name__ == "__main__":
    run_emergent_narrative_ecosystem_demo()
