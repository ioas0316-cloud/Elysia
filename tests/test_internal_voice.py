import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World

class TestInternalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline with a real KGManager and mocked dependencies."""
        self.test_kg_path = Path('data/test_internal_voice_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        # Use a real KGManager for this test to verify KG-based reasoning
        self.kg_manager_instance = KGManager(filepath=str(self.test_kg_path))
        self.kg_manager_instance.add_edge("black hole", "gravity", "related_to")
        self.kg_manager_instance.add_edge("black hole", "celestial body", "is_a")
        self.kg_manager_instance.save()

        # Mock other major dependencies
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_cellular_world = MagicMock(spec=World)
        self.mock_core_memory.get_unasked_hypotheses.return_value = []

        # Add basic behavior to the mocks to prevent errors in handlers
        self.mock_cellular_world.get_cell.return_value = None
        self.mock_cellular_world.run_simulation_step.return_value = []
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5 # Prevent VCD from crashing

        # Instantiate the pipeline with the real KG and mocked components
        self.pipeline = CognitionPipeline(
            kg_manager=self.kg_manager_instance,
            core_memory=self.mock_core_memory,
            wave_mechanics=self.mock_wave_mechanics,
            cellular_world=self.mock_cellular_world
        )

    def tearDown(self):
        """Clean up the test KG file after each test."""
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

    def test_generates_response_from_kg(self):
        """
        Tests that the pipeline can generate a response based on facts in its KG.
        """
        # 1. Process a message that should trigger a KG lookup
        # We patch the styler to get a predictable output without emotional variance
        with patch.object(self.pipeline.entry_handler._successor._successor, 'styler') as mock_styler:
            mock_styler.style_response.side_effect = lambda text, state: f"[Styled] {text}"

            response, _ = self.pipeline.process_message("black hole")

            # 2. Assert that the response contains content derived from the KG
            # The exact sentence can vary, so we check for key concepts.
            self.assertIn("[Styled]", response['text'])
            self.assertTrue(
                "black hole" in response['text'] and ("gravity" in response['text'] or "celestial body" in response['text']),
                f"Response did not contain expected KG facts. Got: {response['text']}"
            )

    def test_response_with_no_relevant_kg_facts(self):
        """
        Tests if the pipeline provides a default response when the KG has no relevant info.
        """
        # 1. Process a message with no relevant memory in the KG
        with patch.object(self.pipeline.entry_handler._successor._successor, 'styler') as mock_styler:
            mock_styler.style_response.side_effect = lambda text, state: f"[Styled] {text}"
            response, _ = self.pipeline.process_message("Tell me about something new.")

            # 2. Assert that the response matches the pipeline's default response
            expected_response = "[Styled] 흥미로운 관점이네요. 조금 더 생각해볼게요."
            self.assertEqual(response['text'], expected_response)


if __name__ == '__main__':
    unittest.main()
