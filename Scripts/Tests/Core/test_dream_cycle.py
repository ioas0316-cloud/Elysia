
import unittest
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Adjust path to find Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from pathlib import Path
from Core.L2_Metabolism.Cycles.dream_protocol import DreamAlchemist

class TestDreamCycle(unittest.TestCase):
    def setUp(self):
        self.alchemist = DreamAlchemist()
        # Mock paths to avoid writing to actual data folder during test
        self.alchemist.queue_path = Path("test_dream_queue.json")
        self.alchemist.wisdom_path = Path("test_crystallized_wisdom.json")

        # Create dummy queue
        dreams = [
            {"intent": "Test Intent", "hypothesis": "Test Hypothesis", "timestamp": "NOW"},
            {"intent": "Weak Idea", "hypothesis": "Blah", "timestamp": "NOW"} # Should be filtered ideally, but our mock resonance is high
        ]
        with open(self.alchemist.queue_path, "w") as f:
            json.dump(dreams, f)

    def tearDown(self):
        if os.path.exists(self.alchemist.queue_path):
            os.remove(self.alchemist.queue_path)
        if os.path.exists(self.alchemist.wisdom_path):
            os.remove(self.alchemist.wisdom_path)

    def test_dream_consolidation(self):
        """Test that dreams are processed and saved to wisdom file."""
        self.alchemist.sleep()

        # Check wisdom file
        self.assertTrue(os.path.exists(self.alchemist.wisdom_path), "Wisdom file not created")
        with open(self.alchemist.wisdom_path, "r") as f:
            wisdom = json.load(f)

        self.assertTrue(len(wisdom) >= 2, "Dreams were not crystallized")
        self.assertEqual(wisdom[0]["intent"], "Test Intent")
        self.assertEqual(wisdom[0]["origin"], "Dream")

        # Check queue is empty
        with open(self.alchemist.queue_path, "r") as f:
            queue = json.load(f)
        self.assertEqual(len(queue), 0, "Dream queue was not cleared")

if __name__ == "__main__":
    unittest.main()
