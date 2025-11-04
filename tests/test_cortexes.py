import unittest
from unittest.mock import patch
import os
import json
from pathlib import Path
from Project_Sophia.value_cortex import ValueCortex
from Project_Sophia.sensory_cortex import SensoryCortex
from Project_Sophia.action_cortex import ActionCortex

class TestCortexes(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.test_kg_path = Path('data/test_kg.json')
        self.backup_kg_path = Path('data/kg.json.bak')
        self.original_kg_path = Path('data/kg.json')

        self.test_tools_kg_path = Path('data/tools_kg.json')
        self.backup_tools_kg_path = Path('data/tools_kg.json.bak')

        if self.original_kg_path.exists():
            self.original_kg_path.rename(self.backup_kg_path)

        if self.test_tools_kg_path.exists():
            self.test_tools_kg_path.rename(self.backup_tools_kg_path)

        dummy_kg = {
            "nodes": [
                {"id": "test_concept", "position": {"x": 0, "y": 0, "z": 0}},
                {"id": "love", "position": {"x": 1, "y": 0, "z": 0}}
            ],
            "edges": [
                {"source": "test_concept", "target": "love", "relation": "is_related_to"}
            ]
        }
        with open(self.test_kg_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_kg, f)

        dummy_tools_kg = {
            "nodes": [
                {"id": "read", "position": {"x": 0, "y": 0, "z": 0}, "activation_energy": 0.0},
                {"id": "file", "position": {"x": 0, "y": 1, "z": 0}, "activation_energy": 0.0},
                {"id": "read_file", "position": {"x": 1, "y": 0, "z": 0}, "activation_energy": 0.0}
            ],
            "edges": [
                {"source": "read", "target": "read_file", "relation": "activates"},
                {"source": "file", "target": "read_file", "relation": "activates"}
            ]
        }
        with open(self.test_tools_kg_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_tools_kg, f)

        self.value_cortex = ValueCortex(kg_path=str(self.test_kg_path))
        self.sensory_cortex = SensoryCortex(self.value_cortex)
        self.action_cortex = ActionCortex()
        self.action_cortex.api_available = True

    def tearDown(self):
        """Tear down after tests."""
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()
        if self.test_tools_kg_path.exists():
            self.test_tools_kg_path.unlink()

        if self.backup_kg_path.exists():
            self.backup_kg_path.rename(str(self.original_kg_path))
        if self.backup_tools_kg_path.exists():
            self.backup_tools_kg_path.rename(str(self.test_tools_kg_path))

    def test_value_cortex_finds_path(self):
        """Test that the ValueCortex can find a path to a core value."""
        path = self.value_cortex.find_meaning_connection("test_concept")
        self.assertEqual(path, ["test_concept", "love"])

    def test_sensory_cortex_visualizes_concept(self):
        """Test that the SensoryCortex can generate an image."""
        image_path = self.sensory_cortex.visualize_concept("test_concept")
        self.assertTrue(os.path.exists(image_path))
        os.remove(image_path)

    @patch('Project_Sophia.action_cortex.ActionCortex._find_best_tool')
    @patch('Project_Sophia.action_cortex.generate_text')
    def test_action_cortex_finds_tool_and_extracts_params(self, mock_generate_text, mock_find_best_tool):
        """
        Test that ActionCortex finds the best tool and extracts its parameters.
        """
        # Mock the tool finding mechanism to always return 'read_file'
        mock_find_best_tool.return_value = 'read_file'
        # Mock the LLM response for parameter extraction
        mock_generate_text.return_value = '```json\n{"filepath": "data/example.txt"}\n```'

        prompt = "Can you read the file 'data/example.txt' for me?"
        action = self.action_cortex.decide_action(prompt)

        self.assertIsNotNone(action, "Action should not be None")
        self.assertEqual(action['tool_name'], 'read_file')
        self.assertIn('filepath', action['parameters'])
        self.assertEqual(action['parameters']['filepath'], 'data/example.txt')

        # Verify that the mocks were called
        mock_find_best_tool.assert_called_once()
        mock_generate_text.assert_called_once()

if __name__ == '__main__':
    unittest.main()
