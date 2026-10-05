#!/usr/bin/env python3
"""
Project Elysia: Standalone 4-Stage Causal Cognitive Loop Demonstration Script
=============================================================================
Demonstrates human/agent cognitive loop simulation:
1. Perception (지각): External reality V_Env(t) projection
2. Discrimination & Judgment (분별과 판단): Dissonance accumulation & Quantum Phase Transition
3. Cognition & Reflection (사고와 반성): Meta-cognitive relaxation & belief calibration
4. Action & Internalizing Causality (행위와 인과 내재화): Action manifestation & macro era reverse projection

Renders Stone Librande's "One-Page Design Dashboard Report" in terminal.
"""

import numpy as np
from elysia_core.causal_cognitive_loop import (
    CausalCognitiveAgent,
    CausalCognitiveEngine
)
from elysia_core.unified_phase_space import (
    ResourceEntity,
    CraftedItemEntity,
    MonsterEntity
)


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: 4-STAGE CAUSAL COGNITIVE LOOP DEMONSTRATION ")
    print("=" * 80 + "\n")

    # 1. Initialize Engine with Era 1: Holy Peace Era
    engine = CausalCognitiveEngine(
        env_name="Phase 1: 신정정치 평화기 (Holy Peace Era)",
        env_base_vec=np.array([-0.4, -0.5, 0.8, 0.8, 0.2])
    )

    # 2. Register Cognitive Agents
    captain = CausalCognitiveAgent(
        name="기사장 칼하인츠",
        base_S=np.array([-0.3, -0.4, 0.7, 0.8, 0.1]),
        dissonance_threshold=0.8,
        relaxation_rate=0.2,
        reverse_projection_weight=0.08
    )

    scholar = CausalCognitiveAgent(
        name="원로학자 이소르",
        base_S=np.array([-0.5, -0.2, 0.9, 0.6, 0.5]),
        dissonance_threshold=0.7,
        relaxation_rate=0.15,
        reverse_projection_weight=0.05
    )

    engine.add_cognitive_agent(captain)
    engine.add_cognitive_agent(scholar)

    # 3. Register Resources, Items & Monsters
    mana_crystal = ResourceEntity("고위상 마나석", np.array([-0.2, 0.3, 0.7, 0.2, 0.6]))
    engine.add_resource(mana_crystal)
    mana_crystal.adapt_to_environment(engine.env.vector)

    holy_blade = CraftedItemEntity("신성응축검", [mana_crystal.current_R], energy_shell_n=2)
    engine.add_crafted_item(holy_blade)

    abyss_boss = MonsterEntity("심연의 파수꾼", np.array([0.4, 0.1, 0.5, -0.2, 0.2]), energy_shell=3)
    engine.add_monster(abyss_boss)

    # Initial Cycle 1
    engine.step_cognitive_cycle()
    print(">>> [CYCLE 1: STABLE ERA] 1차 인지 순환 루프 수행 완료 <<<")
    print(engine.render_cognitive_one_page_dashboard(title_suffix="[ERA 1: PEACE]"))
    print("\n")

    # 4. Cycle 2: Trigger Macro Zeitgeist Shift (War & Chaos Era)
    print(">>> [EVENT] 거대 시대상 포텐셜 대변혁 발생! (전쟁과 오염의 시대 도래) <<<")
    print(">>> [STAGE 1: PERCEPTION] 에이전트들이 외부 환경 자극 V_Env(t)을 실시간 수신함 <<<\n")

    war_event_delta = np.array([1.3, 1.1, -1.5, -1.4, 0.4])
    engine.step_cognitive_cycle(macro_event_delta=war_event_delta)

    print(">>> [STAGE 2 & 3: JUDGMENT & REFLECTION] 불협화음 장력 폭발 및 위상 전이(Phase Transition) 발생! <<<")
    print(">>> [STAGE 4: REVERSE PROJECTION] 에이전트들의 가치관 변화가 다시 거시 시대상(V_Era)에 역투영됨 <<<\n")

    print(engine.render_cognitive_one_page_dashboard(title_suffix="[ERA 2: WAR & CHAOS]"))
    print("\n")

    # 5. Cycle 3: Continuous Relaxation Loop
    engine.step_cognitive_cycle()
    print(">>> [CYCLE 3: RE-STABILIZATION] 메타인지 이완 동역학에 의한 자가 반성 및 가치관 복원 진행 <<<")
    print(engine.render_cognitive_one_page_dashboard(title_suffix="[ERA 2: RELAXATION]"))

    print("\n" + "=" * 80)
    print(" 4-STAGE CAUSAL COGNITIVE LOOP DEMONSTRATION COMPLETE ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
