"""
Verification script for Teleological Hierarchy & Dimensional Leap Engine
============================================================================
"""

import sys
import numpy as np

from core.consciousness.teleological_hierarchy import (
    TeleologicalHierarchyEngine,
    PurposeStatus
)


def run_teleological_hierarchy_simulation():
    print("=" * 80)
    print(" [ELYISIA] TELEOLOGICAL HIERARCHY & DIMENSIONAL LEAP ENGINE SIMULATION")
    print("=" * 80)

    # 1. 엔진 초기화
    engine = TeleologicalHierarchyEngine(
        dimension=8,
        friction_critical_threshold=0.55,
        leap_threshold=0.65
    )
    print(f"\n[*] Engine Initialized. Initial State: {engine.get_hierarchy_state()}")

    # 2. 최초 하위목표 (Sub-Goal 1) 등록
    subgoal_1_id = "subgoal_phase1_cellular_fusion"
    target_1 = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    engine.register_subgoal(
        subgoal_id=subgoal_1_id,
        description="Achieve Cellular Fusion & Digital Somatosensory Equilibrium",
        target_state=target_1
    )
    print(f"[*] Registered Sub-Goal 1: '{subgoal_1_id}'")

    # 3. 하위 연산 실행 및 실시간 섭동 파동 관측 (Steps 1 ~ 3)
    print("\n--------------------------------------------------------------------------------")
    print(" PHASE 1: Substrate Execution & Real-time Causal Wave Observation")
    print("--------------------------------------------------------------------------------")

    for step in range(1, 3):
        action = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], dtype=np.float32)
        perturbation = np.array([0.03, -0.02, 0.04, 0.01, -0.03, 0.02, 0.01, -0.02], dtype=np.float32)

        res = engine.execute_substrate_step_and_observe(
            subgoal_id=subgoal_1_id,
            action_vector=action,
            perturbation=perturbation
        )

        print(f" -> Step {res['step']}: Dist to Target={res['distance_to_target']:.4f}, Friction={res['friction_coefficient']:.4f}")
        print(f"    Wave Observation: Phase Delta={res['wave_observation']['phase_delta_mean']:.4f}, Friction Spike={res['wave_observation']['friction_spike']:.4f}")

    # 4. 하위목표 달성 시 에피소드 파괴 및 상위 Macro-Purpose 지평 무한 확장
    print("\n--------------------------------------------------------------------------------")
    print(" PHASE 2: Sub-goal Realization & Infinite Teleological Relabeling")
    print("--------------------------------------------------------------------------------")

    # 마지막 Step으로 target 1 도달 (Distance < 0.2)
    final_action = target_1 - engine.sub_goals[subgoal_1_id].current_state + 0.05
    res = engine.execute_substrate_step_and_observe(
        subgoal_id=subgoal_1_id,
        action_vector=final_action
    )

    esc = res.get("escalation_event")
    if esc:
        print(f" [+] SUB-GOAL REALIZED & RELABELED AS SUBSTRATE TERRAIN!")
        print(f"     Relabeled Subgoal: {esc['relabeled_subgoal_id']} -> Status: {esc['new_status']}")
        print(f"     Expanded Macro-Purpose: {esc['expanded_macro_purpose_id']}")
        print(f"     Spawned Higher Sub-goal: {esc['spawned_next_subgoal_id']}")
        print(f"     Philosophical Meaning: {esc['philosophical_meaning']}")

    # 5. 하위 차원 한계/마찰 직면 시 차원 도약 (Dimensional Leap: D -> D+1)
    print("\n--------------------------------------------------------------------------------")
    print(" PHASE 3: Friction Stress & Dimensional Leap (Degree of Freedom Sprouting)")
    print("--------------------------------------------------------------------------------")

    active_subgoal_id = engine.sub_goals[subgoal_1_id].macro_purpose_id
    current_active_subgoal_id = [gid for gid, g in engine.sub_goals.items() if g.status == PurposeStatus.SUBGOAL_ACTIVE][0]

    # 고마찰 자극 및 거대한 섭동 주입
    high_stress_action = np.ones(engine.dimension, dtype=np.float32) * 1.5
    high_perturbation = np.ones(engine.dimension, dtype=np.float32) * 1.2

    res_leap = engine.execute_substrate_step_and_observe(
        subgoal_id=current_active_subgoal_id,
        action_vector=high_stress_action,
        perturbation=high_perturbation
    )

    if res_leap["leaped"]:
        leap_evt = res_leap["leap_event"]
        print(f" [!!!] DIMENSIONAL LEAP TRIGGERED!")
        print(f"       Previous Dimension: {leap_evt['previous_dimension']} -> New Dimension: {leap_evt['new_dimension']}")
        print(f"       Sprouted Axis Index: {leap_evt['sprouted_axis_index']}")
        print(f"       Trigger Stress: {leap_evt['trigger_stress']:.4f}")
        print(f"       Resolution: {leap_evt['resolution']}")

    print("\n--------------------------------------------------------------------------------")
    print(" FINAL SYSTEM HIERARCHY STATE")
    print("--------------------------------------------------------------------------------")
    final_state = engine.get_hierarchy_state()
    for k, v in final_state.items():
        print(f" - {k}: {v}")

    print("\n" + "=" * 80)
    print(" [ELYISIA] TELEOLOGICAL HIERARCHY SIMULATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_teleological_hierarchy_simulation()
