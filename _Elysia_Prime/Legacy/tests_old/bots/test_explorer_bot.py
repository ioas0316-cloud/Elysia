# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock, patch

from nano_core.bots.explorer import ExplorerBot
from nano_core.bus import MessageBus
from nano_core.message import Message
from nano_core.registry import ConceptRegistry

class TestExplorerBot(unittest.TestCase):

    def setUp(self):
        """Set up a mock ConceptRegistry and a real MessageBus for each test."""
        self.bus = MessageBus()
        self.mock_registry = MagicMock(spec=ConceptRegistry)
        self.mock_registry.kg = MagicMock()
        self.explorer = ExplorerBot()

    @patch('nano_core.bots.explorer.write_event')
    def test_mission_completion(self, mock_write_event):
        """Test that the bot correctly identifies mission completion."""
        # Setup: The bot receives a message where the current node is the target
        mission_msg = Message(
            verb='explore',
            slots={'start_node': 'A', 'target': 'B', 'path': ['A', 'B']}
        )

        self.explorer.handle(mission_msg, self.mock_registry, self.bus)

        # Verification: A 'mission_complete' event should be written
        mock_write_event.assert_any_call('explorer.mission_complete', {'path': ['A', 'B'], 'target': 'B'})
        # The bus should be empty as the mission is over
        self.assertTrue(self.bus.empty())

    @patch('nano_core.bots.explorer.write_event')
    def test_path_exploration(self, mock_write_event):
        """Test that the bot explores unvisited neighbors."""
        # Setup: KG has a simple path A -> B -> C
        self.mock_registry.kg.get_neighbors.return_value = ['C']

        # Message is at node 'B', heading to 'C'
        mission_msg = Message(
            verb='explore',
            slots={'start_node': 'A', 'target': 'C', 'path': ['A', 'B']},
            strength=0.8
        )

        self.explorer.handle(mission_msg, self.mock_registry, self.bus)

        # Verification: A new 'explore' message should be on the bus for the next step
        self.assertFalse(self.bus.empty())
        next_msg = self.bus.get_next()

        self.assertEqual(next_msg.verb, 'explore')
        self.assertEqual(next_msg.slots['path'], ['A', 'B', 'C'])
        self.assertEqual(next_msg.slots['target'], 'C')
        self.assertAlmostEqual(next_msg.strength, 0.72) # 0.8 * 0.9

        # A step event should also have been logged
        mock_write_event.assert_any_call('explorer.step', {'path': ['A', 'B'], 'current_node': 'B', 'target': 'C'})

if __name__ == '__main__':
    unittest.main()