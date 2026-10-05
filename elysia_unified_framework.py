#!/usr/bin/env python3
"""
Project Elysia: Standalone Unified 5D Phase Space & One-Page Dashboard Demo
=============================================================================
Demonstrates macro Zeitgeist shifts, cascading relaxation across resources, items,
monsters, and NPCs, and renders Stone Librande's "One-Page Design Dashboard".
"""

import numpy as np
from elysia_core.unified_phase_space import (
    ElysiaUnifiedEngine,
    ResourceEntity,
    CraftedItemEntity,
    MonsterEntity,
    NPCEntity
)


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: UNIFIED 5D PHASE SPACE & ONE-PAGE DASHBOARD DEMO ")
    print("=" * 80 + "\n")

    # 1. Initialize Engine with Phase 1: Holy Empire Era
    # Vector: [Will, Flex, Order, Origin, Field]
    engine = ElysiaUnifiedEngine(
        env_name="Phase 1: 신정정치 평화기 (Holy Peace Era)",
        env_base_vec=np.array([-0.4, -0.5, 0.8, 0.8, 0.2])
    )

    # 2. Register Subsystem Entities
    iron_ore = ResourceEntity("철광석", np.array([0.0, 0.4, 0.2, -0.4, -0.2]))
    mandrake = ResourceEntity("만드라고라", np.array([-0.1, -0.3, 0.1, 0.8, 0.2]))
    engine.add_resource(iron_ore)
    engine.add_resource(mandrake)

    # Initial resource adaptation
    iron_ore.adapt_to_environment(engine.env.vector)
    mandrake.adapt_to_environment(engine.env.vector)

    holy_sword = CraftedItemEntity(
        "위상 응축 성검",
        [iron_ore.current_R, mandrake.current_R],
        energy_shell_n=2
    )
    engine.add_crafted_item(holy_sword)

    gargoyle_boss = MonsterEntity("가고일", np.array([0.2, -0.3, 0.4, 0.1, -0.2]), energy_shell=3)
    engine.add_monster(gargoyle_boss)

    npc_captain = NPCEntity("기사장 칼하인츠", np.array([0.1, -0.3, 0.7, 0.8, 0.1]))
    engine.add_npc(npc_captain)

    # Initial relaxation step
    engine.step_cascade()

    # 3. Render Phase 1 One-Page Dashboard Report
    print(engine.render_one_page_dashboard(title_suffix="[PHASE 1: PEACE]"))
    print("\n")

    # 4. Trigger Macro Zeitgeist Shift (War, Chaos & Fire Era)
    # Delta Shift: [Will +1.3, Flex +1.2, Order -1.5, Origin -1.4, Field +0.5]
    print(">>> [EVENT] 거대 시대상 포텐셜 대변혁 발생! (전쟁과 화염/오염의 시대 도래) <<<\n")
    engine.env.name = "Phase 2: 전쟁과 오염의 시대 (War & Chaos Era)"
    engine.step_cascade(delta_env=np.array([1.3, 1.2, -1.5, -1.4, 0.5]))

    # 5. Render Phase 2 One-Page Dashboard Report
    print(engine.render_one_page_dashboard(title_suffix="[PHASE 2: WAR]"))
    print("\n" + "=" * 80)
    print(" UNIFIED 5D PHASE SPACE DEMONSTRATION COMPLETE ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
