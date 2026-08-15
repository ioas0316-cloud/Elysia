"""
Unit tests for Faction Void Gradients & Dynamic Mission Emergence Architecture.
"""

import unittest
from core.causal_world.protocol import Entity, Stats
from core.causal_world.factions import (
    Faction,
    FactionType,
    VoidState,
    DynamicMission,
    VoidGradientSolver
)


class TestFactionsAndVoids(unittest.TestCase):
    def setUp(self):
        # Create Factions
        self.syndicate = Faction(
            faction_id="merchant_syn_01",
            name="Golden Compass Syndicate",
            faction_type=FactionType.MERCHANT_SYNDICATE,
            base_wealth=5000.0,
            initial_voids=VoidState(str_void=40.0, agi_void=50.0) # High security & escort deficiency
        )

        self.mercenaries = Faction(
            faction_id="merc_guild_01",
            name="Iron Fist Mercenaries",
            faction_type=FactionType.MERCENARY_GUILD,
            base_wealth=200.0,
            initial_voids=VoidState(wealth_void=60.0, spr_void=35.0) # High wealth & legitimacy deficiency
        )

        self.factions = {
            self.syndicate.faction_id: self.syndicate,
            self.mercenaries.faction_id: self.mercenaries
        }
        self.solver = VoidGradientSolver(self.factions)

        # Create Player Entity with high STR and AGI
        self.hero = Entity(
            entity_id="hero_01",
            name="Valiant Escort",
            base_stats=Stats(STR=20.0, VIT=15.0, INT=10.0, AGI=18.0, SPR=10.0)
        )

    def test_dynamic_mission_generation(self):
        # Evaluate merchant syndicate missions (STR_void=40, AGI_void=50)
        missions = self.syndicate.evaluate_and_generate_missions(threshold_pressure=30.0)

        self.assertEqual(len(missions), 2)
        target_voids = [m.target_void_type for m in missions]
        self.assertIn("STR", target_voids)
        self.assertIn("AGI", target_voids)

    def test_void_gradient_solver_matching(self):
        self.syndicate.evaluate_and_generate_missions(threshold_pressure=30.0)

        matches = self.solver.match_entity_to_missions(self.hero)
        self.assertGreater(len(matches), 0)

        best_mission, match_score = matches[0]
        self.assertEqual(best_mission.issuing_faction_id, self.syndicate.faction_id)
        self.assertGreater(match_score, 0.0)

    def test_mission_fulfillment_and_void_relief(self):
        missions = self.syndicate.evaluate_and_generate_missions(threshold_pressure=30.0)
        mission_to_do = missions[0]

        initial_wealth = self.hero.wealth
        initial_str_void = self.syndicate.void_state.str_void

        success = self.solver.fulfill_mission(self.hero, mission_to_do)

        self.assertTrue(success)
        self.assertTrue(mission_to_do.is_completed)
        self.assertGreater(self.hero.wealth, initial_wealth)
        # Verify Void was relieved
        self.assertLess(self.syndicate.void_state.str_void, initial_str_void)


if __name__ == "__main__":
    unittest.main()
