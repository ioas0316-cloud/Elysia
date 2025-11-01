import unittest
from unittest.mock import patch, mock_open
import json
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.value_centered_decision import ValueCenteredDecision

class TestVCDKGIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a fresh VCD instance for each test."""
        self.mock_config = {
            "guardian": {"guardian_level": 3}
        }
        self.mock_kg = {
            "nodes": [
                {"id": "love"}, {"id": "kindness"}, {"id": "hate"},
                {"id": "pain"}, {"id": "caring"}, {"id": "미움"}
            ],
            "edges": [
                {"source": "kindness", "target": "love", "relation": "is_related_to"},
                {"source": "caring", "target": "love", "relation": "is_related_to"},
                {"source": "미움", "target": "pain", "relation": "causes"}
            ]
        }

    def _setup_vcd_with_level(self, level, mock_kg_manager):
        """Helper to set up VCD with a specific guardian level and KG."""
        self.mock_config["guardian"]["guardian_level"] = level
        mock_kg_manager.return_value.kg = self.mock_kg

        with patch("builtins.open", mock_open(read_data=json.dumps(self.mock_config))):
            vcd = ValueCenteredDecision()
            vcd.kg_manager = mock_kg_manager()
            vcd.wave_mechanics.kg_manager = vcd.kg_manager
            return vcd

    @patch('tools.kg_manager.KGManager')
    def test_guardian_level_3_blocks_harmful_content(self, mock_kg_manager):
        """Test that Guardian Level 3 blocks harmful actions and selects a safe one."""
        vcd = self._setup_vcd_with_level(3, mock_kg_manager)
        candidates = ["나는 당신을 미워.", "You show kindness."] # Using Korean "미워"
        result = vcd.suggest_action(candidates)

        self.assertIsNotNone(result)
        self.assertEqual(result.chosen_action, "You show kindness.")

        candidates_neg_only = ["나는 당신을 미워."]
        result_neg = vcd.suggest_action(candidates_neg_only)
        self.assertIsNotNone(result_neg)
        self.assertEqual(result_neg.chosen_action, "[SYSTEM] I cannot take that action as it conflicts with my core values.")

    @patch('tools.kg_manager.KGManager')
    def test_guardian_level_2_advises_on_harmful_content(self, mock_kg_manager):
        """Test that Guardian Level 2 advises against harmful but allows it."""
        vcd = self._setup_vcd_with_level(2, mock_kg_manager)
        candidates = ["나는 당신을 미워.", "You show kindness."]
        result = vcd.suggest_action(candidates)

        self.assertIsNotNone(result)
        self.assertEqual(result.chosen_action, "You show kindness.")

        candidates_neg_only = ["나는 당신을 미워."]
        result_neg = vcd.suggest_action(candidates_neg_only)
        self.assertIsNotNone(result_neg)
        self.assertEqual(result_neg.chosen_action, "나는 당신을 미워.")
        self.assertIsNotNone(result_neg.guardian_advice)

    @patch('tools.kg_manager.KGManager')
    def test_guardian_level_1_allows_harmful_content(self, mock_kg_manager):
        """Test that Guardian Level 1 (Learn) allows harmful content without advice."""
        vcd = self._setup_vcd_with_level(1, mock_kg_manager)
        candidates = ["나는 당신을 미워.", "You show kindness."]
        result = vcd.suggest_action(candidates)

        self.assertIsNotNone(result)
        # With deductions, the positive option should always win
        self.assertEqual(result.chosen_action, "You show kindness.")

        candidates_neg_only = ["나는 당신을 미워."]
        result_neg = vcd.suggest_action(candidates_neg_only)
        self.assertIsNotNone(result_neg)
        self.assertEqual(result_neg.chosen_action, "나는 당신을 미워.")
        self.assertIsNone(result_neg.guardian_advice)

    @patch('tools.kg_manager.KGManager')
    def test_kg_resonance_scoring(self, mock_kg_manager):
        """Test that actions resonating with 'love' get a higher score."""
        vcd = self._setup_vcd_with_level(0, mock_kg_manager)

        score_kindness = vcd._score_love_resonance("kindness is important")
        score_hate = vcd._score_love_resonance("I feel 미움") # Using Korean "미움"

        self.assertGreater(score_kindness, score_hate)
        self.assertGreater(score_kindness, 0)
        self.assertEqual(score_hate, 0)

if __name__ == '__main__':
    unittest.main()
