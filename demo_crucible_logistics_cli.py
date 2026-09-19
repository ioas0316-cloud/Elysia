"""
demo_crucible_logistics_cli.py
==============================
CLI Demonstration Runner for Elysia Crucible Equipment Sublimation
and Tiered Logistics Chain Engine.
"""

import time
import json
import numpy as np
from modules.causal_game_engine.crucible_logistics import (
    CausalVector,
    Equipment,
    HeroProfile,
    CrucibleEngine,
    TieredEquipment,
    ResourceTier,
    EquipmentState,
    LogisticsEngine
)
from core.physics.causal_field import CausalField, InformationVoxel


def run_cli_demo():
    print("=" * 80)
    print(" Elysia Causal Game Engine: Crucible & Tiered Logistics Simulation ")
    print(" System: Causal Equipment Sublimation × Graph Supply Chain FSM ")
    print("=" * 80)

    # 1. Initialize Engines
    causal_field = CausalField(dimensions=3)
    crucible_engine = CrucibleEngine(causal_field=causal_field)
    logistics_engine = LogisticsEngine()

    # 2. Register Initial Heroes & Equipment
    hero1 = HeroProfile(
        id="hero_sir_A",
        name="Sir A (3-Star Guard Captain)",
        star_rank=3,
        current_hp_pct=100.0,
        garrison_zone="east_gate",
        equipped_gear=Equipment(id="g_shield", name="Old Iron Shield", memory_pct=0.0)
    )

    hero2 = HeroProfile(
        id="hero_shadow_B",
        name="Shadow Master B (3-Star Rogue)",
        star_rank=3,
        current_hp_pct=100.0,
        garrison_zone="subterranean_market",
        equipped_gear=Equipment(id="g_seal", name="Brass Seal", memory_pct=0.0)
    )

    # Register Hero Voxels into Causal Field
    causal_field.add_voxel(InformationVoxel(
        id=f"hero_{hero1.id}",
        content=hero1.name,
        tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([10.0, 0.0, 0.0], dtype=np.float32)
    ))
    causal_field.add_voxel(InformationVoxel(
        id=f"hero_{hero2.id}",
        content=hero2.name,
        tensor=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, -10.0], dtype=np.float32)
    ))

    # Register Tiered Equipment in Logistics FSM
    magic_barrier = TieredEquipment(
        id="eq_barrier",
        name="High Tower Arcane Barrier",
        tier=ResourceTier.TIER_3_MAGIC,
        magic_aura_shield=50.0
    )
    artifact_seal = TieredEquipment(
        id="eq_seal",
        name="Smuggler's Shadow Seal",
        tier=ResourceTier.TIER_4_ARTIFACT,
        domain_power_enabled=True
    )
    logistics_engine.register_equipment(magic_barrier)
    logistics_engine.register_equipment(artifact_seal)

    print("\n[Initialization Complete]")
    print(f"   - Hero 1: {hero1.name} | Equipped: {hero1.equipped_gear.name}")
    print(f"   - Hero 2: {hero2.name} | Equipped: {hero2.equipped_gear.name}")

    # 3. Simulate Multi-Turn Campaign
    print("\n" + "=" * 80)
    print(" Starting 10-Turn Campaign Simulation...")
    print("=" * 80)

    for turn in range(1, 11):
        print(f"\n>>> [TURN {turn:02d}] --------------")

        # Define turn activities
        if turn < 5:
            # Accumulate CON/STR memory for Sir A in gate defense
            vec_a = CausalVector(w_str=3.0, w_con=8.0)
            crucible_engine.tick_memory_accumulation(hero1, vec_a, environment_bonus=25.0)

            # Accumulate AGI/INT memory for Shadow Master B in black market
            vec_b = CausalVector(w_agi=7.0, w_int=5.0)
            crucible_engine.tick_memory_accumulation(hero2, vec_b, environment_bonus=20.0)

        elif turn == 5:
            # Turn 5: Critical Ordeal Trigger for Sir A (HP drops to 8%, solo defense 6 min)
            hero1.current_hp_pct = 8.0
            hero1.solo_defense_time_min = 6.0
            print("   ⚠️ CRITICAL EVENT: Sir A holds East Gate alone at 8% HP!")

        elif turn == 7:
            # Turn 7: Famine & Black Market Smuggling Ordeal for Shadow Master B
            hero2.famine_active = True
            hero2.citizens_survival_rate = 1.0
            hero2.equipped_gear.memory_pct = 100.0
            print("   ⚠️ CRITICAL EVENT: Severe Famine in Fortress! Shadow Master B smuggles grain via Black Market!")

        elif turn == 8:
            # Turn 8: Enemy Siege blocks Mana Refinery route
            logistics_engine.graph.set_route_blocked(
                "mana_refinery", "mage_tower_3star", blocked=True, cause="Enemy Siege Interception"
            )
            logistics_engine.inventory["mana_crystal"] = 0.0
            print("   ⚠️ SIEGE EVENT: Enemy forces cut off Mana Refinery supply route!")

        # Check equipment sublimation triggers
        sublimed_a = crucible_engine.check_and_trigger_sublimation(hero1, current_turn=turn)
        if sublimed_a:
            print(f"   🔥 [SUBLIMATION EVENT] {hero1.name}'s gear evolved into: {sublimed_a.name}")
            print(f"      - Domain Power: {sublimed_a.domain_power_desc}")

        sublimed_b = crucible_engine.check_and_trigger_sublimation(hero2, current_turn=turn)
        if sublimed_b:
            print(f"   🔥 [SUBLIMATION EVENT] {hero2.name}'s gear evolved into: {sublimed_b.name}")
            print(f"      - Domain Power: {sublimed_b.domain_power_desc}")

        # Tick Logistics Engine
        logistics_out = logistics_engine.tick_logistics_step()

        # Step Causal Field
        causal_field.step(dt=0.1)

        # Print Turn Status Summary
        print(f"   - Hero 1 Gear Memory: {hero1.equipped_gear.memory_pct:.1f}% | Gear: {hero1.equipped_gear.name}")
        print(f"   - Hero 2 Gear Memory: {hero2.equipped_gear.memory_pct:.1f}% | Gear: {hero2.equipped_gear.name}")
        print(f"   - Inventory: Iron Ore={logistics_out['inventory']['iron_ore']:.1f}, "
              f"Mana Crystal={logistics_out['inventory']['mana_crystal']:.1f}, "
              f"Wine={logistics_out['inventory']['fine_wine']:.1f}")
        print(f"   - Arcane Barrier FSM: {logistics_out['fsm_statuses']['eq_barrier']['state']} "
              f"(Aura Shield: {logistics_out['fsm_statuses']['eq_barrier']['magic_aura']:.1f})")
        print(f"   - Strategic Phase: {logistics_out['strategic_phase']}")

    # 4. Display Chronicle History Log
    print("\n" + "=" * 80)
    print(" Chronicle History Log (사서 역사 기록) ")
    print("=" * 80)
    for entry in crucible_engine.chronicle_log.get_all_entries():
        print(entry.to_formatted_string())

    print("\n" + "=" * 80)
    print(" CLI Simulation Completed Successfully! ")
    print("=" * 80)


if __name__ == "__main__":
    run_cli_demo()
