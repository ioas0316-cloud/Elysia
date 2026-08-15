"""
Unit tests for Rumor Propagation & Noise Dynamics Architecture.
"""

import unittest
from core.causal_world.protocol import Entity, Stats
from core.causal_world.propagation import (
    SpatialRegion,
    RumorPacket,
    RumorPropagationNetwork
)


class TestRumorPropagation(unittest.TestCase):
    def setUp(self):
        # Create Regions: Town (origin) and Capital (remote)
        self.village = SpatialRegion("reg_village", "Starter Village", center_pos=(0.0, 0.0), radius=30.0)
        self.capital = SpatialRegion("reg_capital", "Royal Capital", center_pos=(200.0, 0.0), radius=50.0)

        self.network = RumorPropagationNetwork([self.village, self.capital])

        # Merchant NPC who travels between village and capital
        self.merchant_carrier = Entity(
            entity_id="merchant_bob",
            name="Traveling Merchant Bob",
            base_stats=Stats(INT=8.0) # Low INT -> higher exaggeration/noise
        )
        self.merchant_carrier.position = (0.0, 0.0)

    def test_rumor_creation_and_local_state_transition(self):
        # Action in village: Player performs destructive outlaw act
        rumor = self.network.create_rumor_from_action(
            originator_id="player_01",
            target_entity_id="player_01",
            action_type="VILLAGE_RAID",
            reputation_delta=-50.0,
            aggressiveness=1.0,
            origin_pos=(0.0, 0.0),
            timestamp=1.0
        )

        # Check village state transition
        self.assertTrue(self.village.guard_hostility.get("player_01"))
        self.assertTrue(self.village.black_market_unlocked.get("player_01"))
        self.assertFalse(self.village.honored_hero_quests.get("player_01"))

        # Check capital state (rumor hasn't arrived yet)
        self.assertNotIn("player_01", self.capital.entity_perceptions)

    def test_propagation_with_noise_and_remote_transition(self):
        # Create initial rumor in village
        self.network.create_rumor_from_action(
            originator_id="player_01",
            target_entity_id="player_01",
            action_type="VILLAGE_RAID",
            reputation_delta=-30.0,
            aggressiveness=0.8,
            origin_pos=(0.0, 0.0),
            timestamp=1.0
        )

        # Merchant carries rumor from Village (0,0) to Capital (200,0)
        self.network.propagate_and_distort(
            carrier_npc=self.merchant_carrier,
            destination_pos=(200.0, 0.0),
            current_time=10.0,
            rng_seed=42
        )

        # Capital receives rumor with distortion/exaggeration (low INT + long dist)
        perceived_rep = self.capital.entity_perceptions.get("player_01")
        self.assertIsNotNone(perceived_rep)
        # Severity increased due to noise/exaggeration
        self.assertLess(perceived_rep, -30.0)
        # Verify Outlaw state triggered in remote capital due to exaggerated rumor
        self.assertTrue(self.capital.guard_hostility.get("player_01"))


if __name__ == "__main__":
    unittest.main()
