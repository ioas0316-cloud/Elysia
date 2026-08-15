"""
Unit tests for Global Attractor Dynamics Architecture.
"""

import unittest
from core.causal_world.factions import Faction, FactionType, VoidState
from core.causal_world.global_attractor import (
    GlobalAttractorState,
    AttractorType,
    GlobalAttractorEngine
)


class TestGlobalAttractor(unittest.TestCase):
    def setUp(self):
        self.merchants = Faction("f_merchants", "Syndicate", FactionType.MERCHANT_SYNDICATE)
        self.outlaws = Faction("f_outlaws", "Syndicate Outlaws", FactionType.UNDERGROUND_OUTLAWS)

        self.factions = {
            self.merchants.faction_id: self.merchants,
            self.outlaws.faction_id: self.outlaws
        }

        self.engine = GlobalAttractorEngine(self.factions)

        # Register Demon Lord Awakening Attractor
        self.demon_attractor = GlobalAttractorState(
            attractor_type=AttractorType.DEMON_LORD_AWAKENING,
            title="Awakening of Malakor",
            description="Existential threat to all mortal life.",
            current_pressure=50.0,
            threshold_pressure=80.0
        )
        self.engine.register_attractor(self.demon_attractor)

    def test_attractor_accumulation_and_trigger(self):
        # Initially not active and not allied
        self.assertFalse(self.demon_attractor.is_active)
        self.assertFalse(self.engine.is_faction_allied("f_merchants", "f_outlaws"))

        # Add +20 pressure (50 + 20 = 70 < 80)
        triggered = self.engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 20.0)
        self.assertFalse(triggered)

        # Add +15 pressure (70 + 15 = 85 >= 80 -> TRIGGER)
        triggered_now = self.engine.accumulate_world_pressure(AttractorType.DEMON_LORD_AWAKENING, 15.0)
        self.assertTrue(triggered_now)
        self.assertTrue(self.demon_attractor.is_active)

        # Verify emergency survival alliance forced between Merchants and Outlaws
        self.assertTrue(self.engine.is_faction_allied("f_merchants", "f_outlaws"))
        # Verify STR/VIT Void elevated in both factions
        self.assertGreaterEqual(self.merchants.void_state.str_void, 50.0)


if __name__ == "__main__":
    unittest.main()
