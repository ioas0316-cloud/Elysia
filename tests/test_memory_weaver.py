import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.memory_weaver import MemoryWeaver
from Project_Sophia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestMemoryWeaver(unittest.TestCase):

    def setUp(self):
        """Set up mocks for dependencies."""
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)

        # Set up a mock KGManager with the methods we need
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_kg_manager.unlock = MagicMock()
        self.mock_kg_manager.lock = MagicMock()
        self.mock_kg_manager.is_locked = MagicMock(return_value=True)
        self.mock_kg_manager.add_node_if_not_exists = MagicMock()
        self.mock_kg_manager.add_edge = MagicMock()
        self.mock_kg_manager.save_kg = MagicMock()

        # Set up mock KG data
        self.mock_kg_manager.kg = {
            "nodes": [{"id": "learning"}, {"id": "joy"}, {"id": "programming"}],
            "edges": []
        }

        self.weaver = MemoryWeaver(
            core_memory=self.mock_core_memory,
            wave_mechanics=self.mock_wave_mechanics,
            kg_manager=self.mock_kg_manager
        )

    def test_find_related_memories(self):
        """Test that the weaver can find memories with shared concepts."""
        target_memory = {
            "timestamp": "3", "content": "I found joy in learning programming.",
            "emotional_state": {"primary_emotion": "joy"}
        }
        all_memories = [
            {"timestamp": "1", "content": "I am learning about Python."},
            {"timestamp": "2", "content": "I had a bug in my code."},
            target_memory
        ]

        self.mock_core_memory.get_experiences.return_value = all_memories

        related = self.weaver._find_related_memories(target_memory)

        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]['content'], "I am learning about Python.")

    @patch('Project_Sophia.memory_weaver.generate_text')
    def test_synthesize_insight(self, mock_generate_text):
        """Test that a correct prompt is sent to the LLM for insight generation."""
        mock_generate_text.return_value = "Learning new things brings me joy."

        memories = [
            {"timestamp": "1", "content": "I am learning about Python.", "emotional_state": {"primary_emotion": "curiosity"}},
            {"timestamp": "2", "content": "I found joy in learning programming.", "emotional_state": {"primary_emotion": "joy"}}
        ]

        insight = self.weaver._synthesize_insight(memories)

        self.assertEqual(insight, "Learning new things brings me joy.")

        # Verify the prompt construction
        mock_generate_text.assert_called_once()
        call_args = mock_generate_text.call_args
        prompt = call_args[0][0]
        self.assertIn("profound insight", prompt.lower())
        self.assertIn("I am learning about Python.", prompt)
        self.assertIn("(Emotion: joy)", prompt)

    @patch('Project_Sophia.memory_weaver.generate_text')
    def test_update_knowledge_graph(self, mock_generate_text):
        """Test that the KG is correctly updated with a new insight."""
        # Mock the LLM call that parses the insight into a triplet
        mock_generate_text.return_value = "(I, find_joy_in, learning_programming)"

        insight = "I find joy in learning programming."
        self.weaver._update_knowledge_graph(insight)

        # Verify that the KG manager was unlocked and locked
        self.mock_kg_manager.unlock.assert_called_once()
        self.mock_kg_manager.lock.assert_called_once()

        # Verify that nodes and edges were added
        self.mock_kg_manager.add_node_if_not_exists.assert_any_call("I")
        self.mock_kg_manager.add_node_if_not_exists.assert_any_call("learning_programming")
        self.mock_kg_manager.add_edge.assert_called_once_with("I", "learning_programming", "find_joy_in")
        self.mock_kg_manager.save_kg.assert_called_once()

if __name__ == '__main__':
    unittest.main()
