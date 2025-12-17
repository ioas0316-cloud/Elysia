
import unittest
from unittest.mock import MagicMock
import numpy as np
from Core.Foundation.core.genesis_engine import GenesisEngine

class MockWorld:
    def __init__(self):
        self.width = 10
        self.cell_ids = ["hero", "villain"]
        self.positions = np.array([[0, 0, 0], [1, 1, 0]])
        self.ki = np.array([100.0, 50.0])
        self.strength = np.array([10.0, 10.0])
        self.agility = np.array([50.0, 10.0])
        self.wisdom = np.array([30.0, 10.0])
        self.hp = np.array([100.0, 100.0])
        self.max_hp = np.array([100.0, 100.0])
        self.is_injured = np.array([False, False])
        # Mock fields
        self.sunlight_field = np.zeros((10, 10))

class TestGenesisEngine(unittest.TestCase):
    def setUp(self):
        self.world = MockWorld()
        self.engine = GenesisEngine(self.world)

        # Define a sample action via Data (not Code!)
        self.sample_action = {
            "id": "action:piercing_light",
            "type": "action",
            "logic": {
                "cost": {"ki": 20},
                "conditions": [
                    {"check": "stat_ge", "stat": "agility", "value": 40}
                ],
                "effects": [
                    {"op": "damage", "multiplier": 2.0},
                    {"op": "log", "template": "{actor} used Piercing Light on {target}!"}
                ]
            }
        }

        # Load it into the engine
        self.engine.load_definitions({"nodes": [self.sample_action]})

    def test_action_validation_success(self):
        # Hero has 50 agility (>= 40) and 100 ki (>= 20)
        self.assertTrue(self.engine.validate_action(0, "action:piercing_light"))

    def test_action_validation_fail_stat(self):
        # Villain has 10 agility (< 40)
        self.assertFalse(self.engine.validate_action(1, "action:piercing_light"))

    def test_action_validation_fail_cost(self):
        # Drain hero's ki
        self.world.ki[0] = 10
        self.assertFalse(self.engine.validate_action(0, "action:piercing_light"))

    def test_action_execution(self):
        # Hero attacks Villain
        actor_idx = 0
        target_idx = 1

        initial_hp = self.world.hp[target_idx]
        initial_ki = self.world.ki[actor_idx]

        success = self.engine.execute_action(actor_idx, "action:piercing_light", target_idx)

        self.assertTrue(success)
        # Check Cost: 100 - 20 = 80
        self.assertEqual(self.world.ki[actor_idx], 80.0)
        # Check Effect: Damage = 10 * 2.0 = 20. HP = 100 - 20 = 80
        self.assertEqual(self.world.hp[target_idx], 80.0)
        self.assertTrue(self.world.is_injured[target_idx])

if __name__ == '__main__':
    unittest.main()
