"""
Integration & Verification Test Suite for Causal World Architecture
===================================================================
Tests the core principles of Causal World:
1. Universal QWER protocol, equal action execution, resource consumption, and full-loot/destruction on death.
2. Rumor propagation, carrier distortion/noise, and region state transitions (Hero vs Outlaw path).
3. Faction Void Gradients, dynamic mission emergence driven by capability deficiencies, and void relief.
4. Global Attractor pressure accumulation, threshold breach, and emergency survival alliance overrides.
"""

import unittest
import math
from core.causal_world.protocol import (
    Entity, Stats, SkillNode, SkillLayer, SlotType,
    ActionExecutionResult, build_default_layered_skillset
)
from core.causal_world.propagation import (
    RumorPacket, SpatialRegion, RumorPropagationNetwork
)
from core.causal_world.factions import (
    Faction, FactionType, VoidState, DynamicMission, VoidGradientSolver
)
from core.causal_world.global_attractor import (
    GlobalAttractorEngine, GlobalAttractorState, AttractorType
)


class TestCausalWorldIntegration(unittest.TestCase):

    def setUp(self):
        """Set up entities, skillsets, regions, factions, and attractor engines for testing."""
        self.skillset = build_default_layered_skillset()

        # Create Caster (Warrior Player) and Target (Enemy Rogue NPC)
        self.caster = Entity(
            entity_id="player_hero",
            name="Arthur",
            origin="Human",
            bloodline="Swordmaster",
            guild="Order of Knights",
            job="Warrior",
            base_stats=Stats(STR=25.0, VIT=20.0, INT=10.0, AGI=15.0, SPR=15.0)
        )
        self.caster.equip_skill(self.skillset["warrior_heavy_strike"])
        self.caster.equip_skill(self.skillset["swordmaster_pacheonbo"])
        self.caster.equip_skill(self.skillset["knight_salute_mandate"])
        self.caster.equip_skill(self.skillset["human_unity_resonance"])

        self.target = Entity(
            entity_id="bandit_leader",
            name="Malakor",
            origin="Demon",
            bloodline="Shadow Clan",
            guild="Underground Outlaws",
            job="Rogue",
            base_stats=Stats(STR=15.0, VIT=10.0, INT=12.0, AGI=20.0, SPR=8.0)
        )
        self.target.equip_skill(self.skillset["demon_frenzy_protocol"])

    def test_qwer_protocol_execution_and_full_loot(self):
        """Verify QWER skill execution, cooldowns, resource costs, and full loot drop on death."""
        # Arthur executes [Q] Heavy Strike against Malakor
        res1 = self.caster.execute_qwer_action(SlotType.Q, target_entity=self.target, current_time=1.0)
        self.assertTrue(res1.success)
        self.assertGreater(res1.friction_generated, 0.0)
        self.assertLess(self.target.current_hp, self.target.max_hp)

        # Immediate re-cast fails due to cooldown
        res_cd = self.caster.execute_qwer_action(SlotType.Q, target_entity=self.target, current_time=1.2)
        self.assertFalse(res_cd.success)
        self.assertIn("cooldown", res_cd.message.lower())

        # Malakor uses Demon Overclock [R] consuming HP to strike back
        initial_hp = self.target.current_hp
        res_demon = self.target.execute_qwer_action(SlotType.R, target_entity=self.caster, current_time=1.0)
        self.assertTrue(res_demon.success)
        self.assertLess(self.target.current_hp, initial_hp) # HP consumed as cost

        # Reduce Malakor's HP to near zero and land a lethal strike
        self.target.current_hp = 1.0
        res_fatal = self.caster.execute_qwer_action(SlotType.W, target_entity=self.target, current_time=10.0)
        self.assertTrue(res_fatal.success)
        self.assertTrue(self.target.is_dead)
        self.assertEqual(len(self.target.inventory), 0)
        self.assertEqual(self.target.wealth, 0.0)
        self.assertGreater(len(res_fatal.destructed_items), 0) # Items dropped into world

    def test_rumor_propagation_and_region_state_transition(self):
        """Verify rumor packet creation, carrier distortion, and region hero/outlaw path state transitions."""
        capital_region = SpatialRegion(region_id="capital_01", name="Royal Capital", center_pos=(0.0, 0.0), radius=50.0)
        network = RumorPropagationNetwork(regions=[capital_region])

        # Arthur commits a severe outlaw act near capital
        rumor = network.create_rumor_from_action(
            originator_id="player_hero",
            target_entity_id="player_hero",
            action_type="UNPROVOKED_ATTACK",
            reputation_delta=-50.0,
            aggressiveness=0.9,
            origin_pos=(5.0, 5.0),
            timestamp=100.0
        )

        # Verify region immediately updated perception and triggered Outlaw state transition
        self.assertLessEqual(capital_region.entity_perceptions["player_hero"], -40.0)
        self.assertTrue(capital_region.guard_hostility["player_hero"])
        self.assertTrue(capital_region.black_market_unlocked["player_hero"])

        # Low INT carrier transports rumor to remote region with distortion/noise
        frontier_region = SpatialRegion(region_id="frontier_01", name="Frontier Outpost", center_pos=(200.0, 200.0), radius=50.0)
        network.regions["frontier_01"] = frontier_region

        dumb_carrier = Entity(
            entity_id="gossiping_villager",
            name="Bob",
            base_stats=Stats(INT=5.0) # Low INT causes distortion
        )
        dumb_carrier.position = (5.0, 5.0)

        network.propagate_and_distort(
            carrier_npc=dumb_carrier,
            destination_pos=(200.0, 200.0),
            current_time=150.0,
            rng_seed=42
        )

        # Frontier region received rumor with exaggerated severity
        self.assertIn("player_hero", frontier_region.entity_perceptions)
        self.assertLess(frontier_region.entity_perceptions["player_hero"], -50.0)

    def test_faction_void_gradients_and_dynamic_mission_solving(self):
        """Verify Faction Voids generate dynamic missions, solver matches entity, and fulfillment relieves Void."""
        merchants = Faction(
            faction_id="merchant_guild",
            name="Silken Syndicate",
            faction_type=FactionType.MERCHANT_SYNDICATE,
            base_wealth=5000.0,
            initial_voids=VoidState(str_void=60.0, vit_void=40.0) # High security void
        )
        factions = {"merchant_guild": merchants}

        # Generate missions from void pressure
        missions = merchants.evaluate_and_generate_missions(threshold_pressure=30.0)
        self.assertGreater(len(missions), 0)
        str_mission = next(m for m in missions if m.target_void_type == "STR")
        self.assertIn("Caravan Escort", str_mission.title)

        # Solve via VoidGradientSolver
        solver = VoidGradientSolver(factions=factions)
        matches = solver.match_entity_to_missions(self.caster)
        self.assertGreater(len(matches), 0)

        best_mission, match_score = matches[0]
        self.assertEqual(best_mission.mission_id, str_mission.mission_id)

        # Fulfill mission and check void relief
        initial_wealth = self.caster.wealth
        success = solver.fulfill_mission(self.caster, best_mission)
        self.assertTrue(success)
        self.assertGreater(self.caster.wealth, initial_wealth)
        self.assertLess(merchants.void_state.str_void, 60.0) # Void pressure relieved

    def test_global_attractor_dynamics_and_macro_survival_alliance(self):
        """Verify macro attractor pressure accumulation forces emergency alliances across hostile factions."""
        merchants = Faction("f_merchants", "Syndicate", FactionType.MERCHANT_SYNDICATE)
        outlaws = Faction("f_outlaws", "Shadow Syndicate", FactionType.UNDERGROUND_OUTLAWS)
        factions = {"f_merchants": merchants, "f_outlaws": outlaws}

        engine = GlobalAttractorEngine(factions=factions)
        attractor = GlobalAttractorState(
            attractor_type=AttractorType.DEMON_LORD_AWAKENING,
            title="Demon Lord Awakening",
            description="The continent is consumed by abyssal shadow.",
            threshold_pressure=75.0
        )
        engine.register_attractor(attractor)

        # Initial alliance state is False
        self.assertFalse(engine.is_faction_allied("f_merchants", "f_outlaws"))

        # Accumulate world pressure up to breach threshold
        engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 50.0)
        self.assertFalse(attractor.is_active)

        # Breach threshold (50 + 30 = 80 >= 75)
        breached = engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 30.0)
        self.assertTrue(breached)
        self.assertTrue(attractor.is_active)

        # Emergency survival alliance forced between Merchants and Outlaws
        self.assertTrue(engine.is_faction_allied("f_merchants", "f_outlaws"))
        self.assertEqual(merchants.void_state.str_void, 50.0) # Threat elevated security voids


if __name__ == "__main__":
    unittest.main()
