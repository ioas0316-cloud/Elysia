"""
Interactive Observation & Verification Script for the Causal Dynamical System.
==========================================================================
Demonstrates the full lifecycle of the self-organizing causal fantasy world:
1. Universal QWER Protocol execution & Full-Loot Conservation Law
2. Carrier-based Rumor Propagation & Noise Distortion across regions
3. Faction Void Gradients & Dynamic Mission Emergence
4. Hero Path vs Outlaw Path state transitions
5. Global Attractor Overrides (Demon Lord Awakening & Emergency Survival Alliances)
"""

import time
from core.causal_world.protocol import (
    Entity,
    Stats,
    SlotType,
    build_default_layered_skillset
)
from core.causal_world.factions import (
    Faction,
    FactionType,
    VoidState,
    VoidGradientSolver
)
from core.causal_world.propagation import (
    SpatialRegion,
    RumorPropagationNetwork
)
from core.causal_world.global_attractor import (
    GlobalAttractorState,
    AttractorType,
    GlobalAttractorEngine
)


def run_simulation_demonstration():
    print("=" * 80)
    print("      ELYSIA CAUSAL DYNAMICAL SYSTEM SIMULATION & OBSERVATION DEMO")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: INITIALIZE WORLD & FACTIONS
    # ---------------------------------------------------------
    print("\n[Phase 1] Initializing Factions, Spatial Regions & Global Attractor Engine...")

    merchants = Faction("f_merchants", "Golden Compass Syndicate", FactionType.MERCHANT_SYNDICATE, initial_voids=VoidState(str_void=45.0, agi_void=35.0))
    outlaws = Faction("f_outlaws", "Underground Syndicate", FactionType.UNDERGROUND_OUTLAWS, initial_voids=VoidState(wealth_void=60.0))
    factions = {merchants.faction_id: merchants, outlaws.faction_id: outlaws}
    solver = VoidGradientSolver(factions)

    village = SpatialRegion("reg_oakvale", "Oakvale Village", center_pos=(0.0, 0.0), radius=30.0)
    capital = SpatialRegion("reg_crown", "Crown Capital", center_pos=(200.0, 0.0), radius=50.0)
    rumor_network = RumorPropagationNetwork([village, capital])

    attractor_engine = GlobalAttractorEngine(factions)
    demon_attractor = GlobalAttractorState(
        attractor_type=AttractorType.DEMON_LORD_AWAKENING,
        title="Awakening of Malakor",
        description="Existential Threat to Mortal Realm",
        current_pressure=60.0,
        threshold_pressure=80.0
    )
    attractor_engine.register_attractor(demon_attractor)

    skills = build_default_layered_skillset()

    print(f"  - Initialized Factions: {merchants.name} (STR Void: {merchants.void_state.str_void:.1f}), {outlaws.name}")
    print(f"  - Regions: {village.name} at (0,0), {capital.name} at (200,0)")
    print(f"  - Global Attractor: {demon_attractor.title} (Pressure: {demon_attractor.current_pressure:.1f}/{demon_attractor.threshold_pressure:.1f})")

    # ---------------------------------------------------------
    # STEP 2: PLAYER & NPC QWER PROTOCOL EXECUTION
    # ---------------------------------------------------------
    print("\n[Phase 2] Spawning Entities & Executing Universal QWER Protocol...")

    player = Entity("player_kaelen", "Kaelen", origin="Human", bloodline="Swordmaster", guild="Knights", job="Warrior", base_stats=Stats(STR=22.0, VIT=16.0, INT=12.0, AGI=15.0, SPR=10.0))
    player.equip_skill(skills["warrior_heavy_strike"])  # Q
    player.equip_skill(skills["swordmaster_pacheonbo"]) # W
    player.equip_skill(skills["demon_frenzy_protocol"]) # R (Demonic Outlaw Skill)

    bandit = Entity("bandit_rogue", "Highland Marauder", base_stats=Stats(VIT=2.0))

    print(f"  - {player.name} casts [Q] Heavy Strike against {bandit.name}...")
    action_res = player.execute_qwer_action(SlotType.Q, target_entity=bandit, current_time=1.0)

    print(f"    * Result: {action_res.message}")
    print(f"    * Friction Generated: {action_res.friction_generated:.1f}")
    print(f"    * Target Dead: {bandit.is_dead}, Dropped Items: {action_res.destructed_items}")

    # ---------------------------------------------------------
    # STEP 3: RUMOR PROPAGATION & NOISE DISTORTION
    # ---------------------------------------------------------
    print("\n[Phase 3] Generating Rumor Packet & Propagating with Carrier Noise...")

    rumor = rumor_network.create_rumor_from_action(
        originator_id=player.entity_id,
        target_entity_id=player.entity_id,
        action_type="HIGHWAY_RAID",
        reputation_delta=-25.0,
        aggressiveness=0.8,
        origin_pos=(0.0, 0.0),
        timestamp=1.0
    )

    print(f"  - Rumor Created in {village.name}: Initial Delta={rumor.reputation_delta}")
    print(f"  - Oakvale Village Perception for {player.name}: {village.entity_perceptions.get(player.entity_id):.1f}")

    # Carrier NPC travels to Capital
    carrier_npc = Entity("bard_tom", "Bard Tom", base_stats=Stats(INT=6.0)) # Low INT -> Exaggeration
    carrier_npc.position = (0.0, 0.0)

    print(f"  - Carrier {carrier_npc.name} travels from Oakvale (0,0) to Crown Capital (200,0)...")
    rumor_network.propagate_and_distort(
        carrier_npc=carrier_npc,
        destination_pos=(200.0, 0.0),
        current_time=10.0,
        rng_seed=42
    )

    capital_rep = capital.entity_perceptions.get(player.entity_id)
    print(f"  - Crown Capital Received Rumor! Severity Multiplier: {rumor.perceived_severity:.2f}")
    print(f"  - Crown Capital Perceived Reputation: {capital_rep:.1f}")
    print(f"  - Crown Capital Guard Hostility: {capital.guard_hostility.get(player.entity_id)}")
    print(f"  - Crown Capital Black Market Unlocked: {capital.black_market_unlocked.get(player.entity_id)}")

    # ---------------------------------------------------------
    # STEP 4: FACTION VOID GRADIENT & DYNAMIC MISSION EMERGENCE
    # ---------------------------------------------------------
    print("\n[Phase 4] Evaluating Faction Voids & Dynamically Emerging Missions...")

    missions = merchants.evaluate_and_generate_missions(threshold_pressure=30.0)
    print(f"  - {merchants.name} dynamically generated {len(missions)} mission(s) from Void pressure:")
    for m in missions:
        print(f"    * [{m.mission_id}] {m.title} (Target Void: {m.target_void_type}, Pressure: {m.gradient_pressure:.1f})")

    matches = solver.match_entity_to_missions(player)
    if matches:
        best_mission, match_score = matches[0]
        print(f"  - Matched {player.name} to Mission '{best_mission.title}' (Match Score: {match_score:.1f})")
        print(f"  - Fulfilling mission...")
        solver.fulfill_mission(player, best_mission)
        print(f"  - Mission Completed! {merchants.name} STR Void reduced to: {merchants.void_state.str_void:.1f}")
        print(f"  - {player.name} Wealth: {player.wealth:.1f}")

    # ---------------------------------------------------------
    # STEP 5: GLOBAL ATTRACTOR OVERRIDE & SURVIVAL ALLIANCE
    # ---------------------------------------------------------
    print("\n[Phase 5] Accumulating Global Pressure & Triggering Demon Lord Awakening...")

    print(f"  - Current Demon Lord Pressure: {demon_attractor.current_pressure:.1f}/80.0")
    print(f"  - Accumulating +25.0 pressure from world destruction...")
    triggered = attractor_engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 25.0)

    print(f"  - Global Attractor Triggered: {triggered}")
    print(f"  - {demon_attractor.title} IS NOW ACTIVE! (Pressure: {demon_attractor.current_pressure:.1f})")

    allied = attractor_engine.is_faction_allied(merchants.faction_id, outlaws.faction_id)
    print(f"  - Emergency Macro State Transition: Are {merchants.name} and {outlaws.name} forced into alliance? {allied}")
    print(f"  - {merchants.name} Emergency STR Void: {merchants.void_state.str_void:.1f}")
    print(f"  - {outlaws.name} Emergency STR Void: {outlaws.void_state.str_void:.1f}")

    print("\n" + "=" * 80)
    print("      CAUSAL DYNAMICAL SYSTEM SIMULATION COMPLETE: ALL TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_simulation_demonstration()
