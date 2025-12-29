import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.FoundationLayer.Foundation.exploration_cortex import ExplorationCortex
from tools.kg_manager import KGManager
from nano_core.bus import MessageBus

class TestExplorationCortex(unittest.TestCase):
    def setUp(self):
        self.kg_manager = MagicMock(spec=KGManager)
        self.bus = MagicMock(spec=MessageBus)
        self.explorer = ExplorationCortex(self.kg_manager, self.bus)

    def test_launch_exploration_mission_posts_message(self):
        """
        Verify that launching a mission posts a correctly formatted 'explore' message to the bus.
        """
        # 1. Setup mock KG data
        self.kg_manager.kg = {
            "nodes": [
                {"id": "StarA", "activation_energy": 0.8}, # Important node
                {"id": "PlanetX"} # Lonely node
            ],
            "edges": []
        }

        # 2. Run the mission launch
        self.explorer.launch_exploration_mission(num_missions=1)

        # 3. Verify that bus.post was called
        self.bus.post.assert_called_once()
        posted_message = self.bus.post.call_args[0][0]

        # 4. Check the content of the message
        self.assertEqual(posted_message.verb, "explore")
        self.assertEqual(posted_message.src, "ExplorationCortex")
        self.assertIn('start_node', posted_message.slots)
        self.assertEqual(posted_message.slots['start_node'], 'StarA')
        self.assertIn('target', posted_message.slots)
        self.assertEqual(posted_message.slots['target'], 'PlanetX')
        self.assertIn('path', posted_message.slots)
        self.assertEqual(posted_message.slots['path'], ['StarA'])


    def test_generate_definitional_questions(self):
        """
        Test that the cortex generates definitional questions for lonely nodes.
        """
        self.kg_manager.kg = {
            "nodes": [
                {"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}
            ],
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"}
            ]
        }

        questions = self.explorer.generate_definitional_questions(num_questions=1)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0], "What is 'D'?")

if __name__ == "__main__":
    unittest.main()
