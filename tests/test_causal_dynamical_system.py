"""
Comprehensive Integration Tests for the Causal Dynamical System Architecture.
=============================================================================
Verifies all 5 interconnected systems:
1. Universal QWER Protocol & Layered Skill Trees
2. Faction Void Gradients & Dynamic Mission Emergence
3. Rumor Propagation & Carrier Noise/Distortion
4. Conservation Laws & Death/Destruction Void Creation
5. Global Attractor Overrides (Demon Lord Awakening & Survival Alliances)
"""

import unittest
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


class TestCausalDynamicalSystemIntegration(unittest.TestCase):
    def setUp(self):
        # 1. Initialize Default Skillset
        self.skills = build_default_layered_skillset()

        # 2. Setup Factions
        self.merchants = Faction("f_merchants", "Golden Compass Syndicate", FactionType.MERCHANT_SYNDICATE, initial_voids=VoidState(str_void=45.0))
        self.outlaws = Faction("f_outlaws", "Underground Syndicate", FactionType.UNDERGROUND_OUTLAWS, initial_voids=VoidState(wealth_void=50.0))
        self.factions = {
            self.merchants.faction_id: self.merchants,
            self.outlaws.faction_id: self.outlaws
        }
        self.solver = VoidGradientSolver(self.factions)

        # 3. Setup Regions & Rumor Network
        self.starter_village = SpatialRegion("reg_village", "Oakvale Village", center_pos=(0.0, 0.0), radius=30.0)
        self.royal_capital = SpatialRegion("reg_capital", "Crown Capital", center_pos=(200.0, 0.0), radius=50.0)
        self.rumor_network = RumorPropagationNetwork([self.starter_village, self.royal_capital])

        # 4. Setup Global Attractor Engine
        self.attractor_engine = GlobalAttractorEngine(self.factions)
        self.demon_attractor = GlobalAttractorState(
            attractor_type=AttractorType.DEMON_LORD_AWAKENING,
            title="Demon Lord Awakening",
            description="Existential Threat",
            current_pressure=60.0,
            threshold_pressure=80.0
        )
        self.attractor_engine.register_attractor(self.demon_attractor)

        # 5. Setup Entities
        self.player = Entity("player_01", "Kaelen", base_stats=Stats(STR=20.0, VIT=15.0, INT=12.0, AGI=15.0, SPR=10.0))
        self.player.equip_skill(self.skills["warrior_heavy_strike"])  # Q
        self.player.equip_skill(self.skills["swordmaster_pacheonbo"]) # W

        self.carrier_npc = Entity("npc_carrier_01", "Bard Tom", base_stats=Stats(INT=8.0)) # Low INT -> Exaggeration
        self.carrier_npc.position = (0.0, 0.0)

        self.bandit_target = Entity("bandit_01", "Bandit Leader", base_stats=Stats(VIT=2.0))
        self.bandit_target.position = (0.0, 0.0)

    def test_full_end_to_end_causal_chain(self):
        # Step A: Player executes QWER action (Heavy Strike) destroying bandit target
        action_res = self.player.execute_qwer_action(SlotType.Q, target_entity=self.bandit_target, current_time=1.0)
        self.assertTrue(action_res.success)
        self.assertTrue(self.bandit_target.is_dead)
        self.assertGreater(len(action_res.destructed_items), 0)

        # Step B: Rumor generated from destructive action at village (0,0)
        rumor = self.rumor_network.create_rumor_from_action(
            originator_id=self.player.entity_id,
            target_entity_id=self.player.entity_id,
            action_type="BANDIT_EXECUTED",
            reputation_delta=action_res.reputation_delta, # -2.0
            aggressiveness=action_res.aggressiveness,
            origin_pos=(0.0, 0.0),
            timestamp=1.0
        )
        self.assertIsNotNone(rumor)

        # Step C: Bard Tom propagates rumor from village to remote Royal Capital
        self.rumor_network.propagate_and_distort(
            carrier_npc=self.carrier_npc,
            destination_pos=(200.0, 0.0),
            current_time=10.0,
            rng_seed=123
        )
        # Verify distorted perception received in Royal Capital
        capital_perception = self.royal_capital.entity_perceptions.get(self.player.entity_id)
        self.assertIsNotNone(capital_perception)

        # Step D: Merchant Syndicate evaluates Voids -> Player fulfills dynamic mission
        missions = self.merchants.evaluate_and_generate_missions(threshold_pressure=30.0)
        self.assertGreater(len(missions), 0)
        str_mission = [m for m in missions if m.target_void_type == "STR"][0]

        initial_merchant_void = self.merchants.void_state.str_void
        success = self.solver.fulfill_mission(self.player, str_mission)
        self.assertTrue(success)
        # Verify merchant Void relieved by player capability
        self.assertLess(self.merchants.void_state.str_void, initial_merchant_void)

        # Step E: Accumulate world pressure -> Triggers Demon Lord Awakening
        # Death & Void accumulation raises global attractor pressure (+25.0)
        triggered = self.attractor_engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 25.0)
        self.assertTrue(triggered)
        self.assertTrue(self.demon_attractor.is_active)

        # Verify emergency survival alliance forced between Merchants and Outlaws
        self.assertTrue(self.attractor_engine.is_faction_allied(self.merchants.faction_id, self.outlaws.faction_id))


if __name__ == "__main__":
    unittest.main()
