import unittest
from unittest.mock import patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory_base import Memory
from Project_Sophia.emotional_engine import EmotionalState
from tools import kg_manager

class TestInternalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline and a test KG for each test."""
        from pathlib import Path
        self.test_kg_path = Path('data/test_internal_voice_kg.json')
        if self.test_kg_path.exists():
            self.test_kg_path.unlink()

        # Create a KGManager instance specifically for this test
        self.kg_manager_instance = kg_manager.KGManager(filepath=self.test_kg_path)

        # Set up a test KG
        self.kg_manager_instance.add_edge("black hole", "gravity", "related_to")
        self.kg_manager_instance.add_edge("black hole", "celestial body", "is_a")

        # IMPORTANT: Add mock embeddings, as WaveMechanics requires them to spread activation.
        for node_id, mock_embedding in [
            ("black hole", [0.1] * 8),
            ("gravity", [0.2] * 8),
            ("celestial body", [0.3] * 8),
        ]:
            node = self.kg_manager_instance.get_node(node_id)
            if node:
                node['embedding'] = mock_embedding

        self.kg_manager_instance.save()

        self.pipeline = CognitionPipeline()
        self.pipeline.kg_manager = self.kg_manager_instance
        self.pipeline.api_available = False # Ensure we are testing the internal response

    def tearDown(self):
        """Clean up the test KG file after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)


    def test_response_with_no_memory(self):
        """
        Tests if the pipeline provides a learning message when no relevant memory is found.
        """
        # 1. Process a message with no relevant memory in the KG
        response, _ = self.pipeline.process_message("Tell me about something new.")

        # 2. Assert that the response indicates a willingness to learn
        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)

if __name__ == '__main__':
    unittest.main()
