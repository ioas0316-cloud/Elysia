import unittest
from unittest.mock import patch
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory_base import Memory
from Project_Sophia.emotional_engine import EmotionalState
from tools.kg_manager import KGManager # Corrected import
from Project_Sophia.logical_reasoner import LogicalReasoner
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Elysia.value_centered_decision import VCD


class TestInternalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline and a test KG for each test."""
        self.test_kg_path = Path('data/test_internal_voice_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        self.kg_manager_instance = KGManager(filepath=self.test_kg_path)
        self.kg_manager_instance.add_node("black hole", properties={'embedding': [0.1]*8})
        self.kg_manager_instance.add_node("gravity", properties={'embedding': [0.2]*8})
        self.kg_manager_instance.add_node("celestial body", properties={'embedding': [0.3]*8})
        self.kg_manager_instance.add_node("love", properties={'embedding': [0.9]*8})
        self.kg_manager_instance.add_edge("black hole", "gravity", "related_to")
        self.kg_manager_instance.add_edge("black hole", "celestial body", "is_a")
        self.kg_manager_instance.save()

        self.pipeline = CognitionPipeline()
        self.pipeline.kg_manager = KGManager(filepath=self.test_kg_path)
        self.pipeline.reasoner = LogicalReasoner(kg_manager=self.pipeline.kg_manager)
        self.pipeline.wave_mechanics = WaveMechanics(self.pipeline.kg_manager)
        self.pipeline.vcd = VCD(self.pipeline.kg_manager, self.pipeline.wave_mechanics)

    def tearDown(self):
        """Clean up the test KG file after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)

    def test_response_for_unknown_concept_generates_default_message(self):
        """
        Tests that the pipeline provides the new default response for unknown concepts.
        """
        response, _ = self.pipeline.process_message("Tell me about something new.")
        expected_response = "저는 이렇게 생각해요: 흥미로운 관점이네요. 조금 더 생각해볼게요."
        self.assertEqual(response['text'], expected_response)

    def test_response_for_known_concept_contains_deduced_facts(self):
        """
        Tests that a query about a known concept generates a response containing
        facts from the KG, processed through the VCD and synthesizer.
        """
        response, _ = self.pipeline.process_message("Tell me about a black hole.")
        self.assertIn("black hole", response['text'].lower())
        self.assertTrue(
            "gravity" in response['text'].lower() or
            "celestial body" in response['text'].lower()
        )
        self.assertTrue(response['text'].startswith("저는 이렇게 생각해요:"))


if __name__ == '__main__':
    unittest.main()
