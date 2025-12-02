# [Genesis: 2025-12-02] Purified by Elysia

import unittest
import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import Experience

class TestLLMEnhancedDreaming(unittest.TestCase):
    def setUp(self):
        self.mock_core_memory = MagicMock()
        self.spiderweb = Spiderweb()

    @patch('Project_Sophia.dreaming_cortex.generate_text')
    def test_llm_concept_extraction(self, mock_generate_text):
        # Mock LLM response
        mock_generate_text.return_value = """{
            "concepts": ["fire", "heat", "danger"],
            "relations": [
                {"source": "fire", "target": "heat", "type": "causes", "weight": 0.9},
                {"source": "fire", "target": "danger", "type": "enables", "weight": 0.7}
            ]
        }"""

        dreaming_cortex = DreamingCortex(self.mock_core_memory, self.spiderweb, use_llm=True)

        # Setup mock experience
        exp = Experience(
            timestamp="2023-10-27T10:00:00",
            content="I saw a fire burning and it was very hot and dangerous.",
            type="episode"
        )

        self.mock_core_memory.get_unprocessed_experiences.return_value = [exp]

        # Run dream
        dreaming_cortex.dream()

        # Verify concepts were added
        self.assertTrue(self.spiderweb.graph.has_node("fire"))
        self.assertTrue(self.spiderweb.graph.has_node("heat"))
        self.assertTrue(self.spiderweb.graph.has_node("danger"))

        # Verify causal relation was added
        self.assertTrue(self.spiderweb.graph.has_edge("fire", "heat"))
        edge_data = self.spiderweb.graph.get_edge_data("fire", "heat")
        self.assertEqual(edge_data["relation"], "causes")
        self.assertEqual(edge_data["weight"], 0.9)

    def test_fallback_to_naive(self):
        """Test that system falls back to naive mode if LLM fails"""
        dreaming_cortex = DreamingCortex(self.mock_core_memory, self.spiderweb, use_llm=False)

        exp = Experience(
            timestamp="2023-10-27T10:00:00",
            content="The fire was burning",
            type="episode"
        )

        self.mock_core_memory.get_unprocessed_experiences.return_value = [exp]

        # Run dream in naive mode
        dreaming_cortex.dream()

        # Should still extract basic concepts
        self.assertTrue(self.spiderweb.graph.has_node("fire"))
        self.assertTrue(self.spiderweb.graph.has_node("burning"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()