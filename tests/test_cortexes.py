import unittest
from unittest.mock import patch
import os
import json
from pathlib import Path
from Project_Mirror.sensory_cortex import SensoryCortex
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

        self.sensory_cortex = SensoryCortex()
        self.action_cortex = ActionCortex()

    def tearDown(self):
        """Tear down after tests."""
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()
        if self.test_tools_kg_path.exists():
            self.test_tools_kg_path.unlink()

        if self.backup_kg_path.exists():
            self.backup_kg_path.rename(self.original_kg_path)
        if self.backup_tools_kg_path.exists():
            self.backup_tools_kg_path.rename(self.test_tools_kg_path)

    def test_sensory_cortex_visualizes_concept(self):
        """Test that the SensoryCortex can generate an image."""
        image_path = self.sensory_cortex.visualize_concept("test_concept")
        self.assertTrue(os.path.exists(image_path))
        os.remove(image_path)

    @patch('Project_Sophia.action_cortex.generate_text')
    def test_action_cortex_finds_tool_and_extracts_params(self, mock_generate_text):
        """
        Test that ActionCortex finds the best tool and extracts its parameters.
        """
        # Mock the LLM response for parameter extraction
        mock_generate_text.return_value = '```json\n{"filepath": "data/example.txt"}\n```'

        prompt = "Can you read the file 'data/example.txt' for me?"
        action = self.action_cortex.decide_action(prompt)

        self.assertIsNotNone(action)
        self.assertEqual(action['tool_name'], 'read_file')
        self.assertIn('filepath', action['parameters'])
        self.assertEqual(action['parameters']['filepath'], 'data/example.txt')

        # Verify that the mock was called
        mock_generate_text.assert_called_once()

if __name__ == '__main__':
    unittest.main()
