"""
Unit tests for Universal QWER Action Protocol & Layered Skill Tree Architecture.
"""

import unittest
from core.causal_world.protocol import (
    Entity,
    Stats,
    SlotType,
    SkillLayer,
    SkillNode,
    build_default_layered_skillset,
    ActionExecutionResult
)


class TestQWERProtocol(unittest.TestCase):
    def setUp(self):
        self.skills = build_default_layered_skillset()

        # Create Player Entity
        self.player = Entity(
            entity_id="player_01",
            name="Hero Heroic",
            origin="Human",
            bloodline="Commoner",
            guild="Knights",
            job="Warrior",
            base_stats=Stats(STR=15.0, VIT=15.0, INT=10.0, AGI=10.0, SPR=10.0)
        )
        # Equip default QWER skills
        self.player.equip_skill(self.skills["warrior_heavy_strike"])      # Q
        self.player.equip_skill(self.skills["swordmaster_pacheonbo"])     # W
        self.player.equip_skill(self.skills["knight_salute_mandate"])     # E
        self.player.equip_skill(self.skills["human_unity_resonance"])     # R

        # Create Target NPC
        self.target = Entity(
            entity_id="npc_bandit_01",
            name="Rough Bandit",
            origin="Human",
            bloodline="Commoner",
            guild="Outlaws",
            job="Rogue",
            base_stats=Stats(STR=10.0, VIT=8.0, INT=8.0, AGI=12.0, SPR=8.0)
        )

    def test_qwer_action_execution_success(self):
        result = self.player.execute_qwer_action(SlotType.Q, target_entity=self.target, current_time=1.0)

        self.assertTrue(result.success)
        self.assertEqual(result.slot, SlotType.Q)
        self.assertGreater(result.friction_generated, 0.0)
        self.assertLess(self.target.current_hp, self.target.max_hp)

    def test_cooldown_enforcement(self):
        # First execution at t=1.0
        res1 = self.player.execute_qwer_action(SlotType.Q, target_entity=self.target, current_time=1.0)
        self.assertTrue(res1.success)

        # Immediate re-execution at t=1.2 (cooldown is 1.0s)
        res2 = self.player.execute_qwer_action(SlotType.Q, target_entity=self.target, current_time=1.2)
        self.assertFalse(res2.success)
        self.assertIn("cooldown", res2.message)

    def test_reputation_and_outlaw_transition(self):
        # Create a Demon Outlaw player who casts dark skills
        demon_player = Entity(
            entity_id="demon_lord_01",
            name="Lord Malakor",
            origin="Demon",
            bloodline="Shadow Clan",
            guild="Underground",
            job="Warlock"
        )
        demon_player.equip_skill(self.skills["demon_frenzy_protocol"])

        self.assertFalse(demon_player.is_outlaw)

        # Cast R skill twice (reputation_shift is -30.0 each cast)
        res1 = demon_player.execute_qwer_action(SlotType.R, current_time=1.0)
        self.assertTrue(res1.success)
        self.assertEqual(demon_player.reputation_score, -30.0)
        self.assertFalse(demon_player.is_outlaw)

        res2 = demon_player.execute_qwer_action(SlotType.R, current_time=15.0)  # past cooldown
        self.assertTrue(res2.success)
        self.assertEqual(demon_player.reputation_score, -60.0)
        self.assertTrue(demon_player.is_outlaw)

    def test_full_loot_and_destruction_law(self):
        # Target with 10 HP
        weak_target = Entity(
            entity_id="weak_target_01",
            name="Weak Goblin",
            base_stats=Stats(VIT=1.0)
        )
        self.assertFalse(weak_target.is_dead)

        # High damage attack
        result = self.player.execute_qwer_action(SlotType.W, target_entity=weak_target, current_time=1.0)

        self.assertTrue(result.success)
        self.assertTrue(weak_target.is_dead)
        self.assertGreater(len(result.destructed_items), 0)
        self.assertEqual(weak_target.wealth, 0.0)


if __name__ == "__main__":
    unittest.main()
