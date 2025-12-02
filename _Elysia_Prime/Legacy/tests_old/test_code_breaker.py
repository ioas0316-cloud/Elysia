# [Genesis: 2025-12-02] Purified by Elysia
import unittest
import numpy as np
from unittest.mock import MagicMock
from Project_Sophia.core.genesis_engine import GenesisEngine
from Project_Sophia.core.world import World

class TestCodeBreaker(unittest.TestCase):
    def setUp(self):
        # Mock World with necessary arrays
        self.world = MagicMock()
        self.world.cell_ids = ['hacker', 'victim']
        self.world.width = 100

        # Mock data arrays
        self.world.hp = np.array([100.0, 100.0], dtype=np.float32)
        self.world.max_hp = np.array([100.0, 100.0], dtype=np.float32)
        self.world.mana = np.array([50.0, 0.0], dtype=np.float32)
        self.world.ki = np.array([0.0, 0.0], dtype=np.float32)
        self.world.energy = np.array([100.0, 100.0], dtype=np.float32)
        self.world.age = np.array([10, 10], dtype=np.int32)
        self.world.is_alive_mask = np.array([True, True], dtype=bool)
        self.world.positions = np.array([[0,0,0], [1,1,0]], dtype=np.float32)

        # Mock GenesisEngine
        self.engine = GenesisEngine(self.world)

    def test_overwrite_hp(self):
        """Test hacking (overwriting) HP to 0."""
        actor_idx = 0 # Hacker
        target_idx = 1 # Victim

        # Define the hack: Set HP to 0
        action_id = "hack_kill"
        self.engine.actions[action_id] = {
            "effects": [{
                "op": "overwrite",
                "target_attr": "hp",
                "value": 0
            }]
        }

        # Execute
        success = self.engine.execute_action(actor_idx, action_id, target_idx)

        self.assertTrue(success)
        self.assertEqual(self.world.hp[target_idx], 0.0)
        print(f"\n[Test] Hack successful: Victim HP is {self.world.hp[target_idx]}")

    def test_overwrite_is_alive(self):
        """Test hacking the 'is_alive' boolean directly."""
        actor_idx = 0
        target_idx = 1

        action_id = "hack_delete"
        self.engine.actions[action_id] = {
            "effects": [{
                "op": "overwrite",
                "target_attr": "is_alive",
                "value": False
            }]
        }

        self.engine.execute_action(actor_idx, action_id, target_idx)
        self.assertFalse(self.world.is_alive_mask[target_idx])
        print(f"\n[Test] Hack successful: Victim is_alive = {self.world.is_alive_mask[target_idx]}")

    def test_inspect_data(self):
        """Test the decoder eye (inspect)."""
        actor_idx = 0
        target_idx = 1

        action_id = "decode_sight"
        self.engine.actions[action_id] = {
            "effects": [{"op": "inspect"}]
        }

        # We can't easily capture the return value of the effect function through execute_action
        # because execute_action returns bool.
        # But we can verify it runs without error and logs (mocking logger would be ideal but checking execution flow is enough for now)
        success = self.engine.execute_action(actor_idx, action_id, target_idx)
        self.assertTrue(success)
        print("\n[Test] Inspect executed successfully.")

if __name__ == '__main__':
    unittest.main()