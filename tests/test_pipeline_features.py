import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core_memory import Memory, EmotionalState

class TestPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline for each test."""
        # Ensure a clean memory file for each test
        self.memory_path = 'Elysia_Input_Sanctum/test_elysia_core_memory.json'
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)

        self.pipeline = CognitionPipeline()
        # Point the pipeline's memory to a test-specific file
        self.pipeline.core_memory.file_path = self.memory_path

    def tearDown(self):
        """Clean up the memory file after each test."""
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)

    def test_conversational_memory_is_retrieved(self):
        """
        Tests if the pipeline can retrieve a relevant past experience and use
        it in a response.
        """
        # 1. Add a relevant memory to the core memory
        past_experience = Memory(
            timestamp="2025-01-01T12:00:00",
            content="I enjoy learning about black holes.",
            emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])
        )
        self.pipeline.core_memory.add_experience(past_experience)

        # 2. Ask a question related to the past experience
        response, _, _ = self.pipeline.process_message("What do you know about black holes?")

        # 3. Assert that the response references the past conversation
        self.assertIn("이전에 'I enjoy learning about black holes.'에 대해 이야기 나눈 것을 기억해요.", response)

    @patch('Project_Sophia.inquisitive_mind.generate_text')
    def test_inquisitive_mind_is_triggered(self, mock_generate_text):
        """
        Tests if the InquisitiveMind is triggered when the pipeline encounters
        a question it cannot answer from memory or internal knowledge.
        """
        # 1. Mock the external LLM call to avoid actual API usage
        mock_response = "A supermassive black hole is the largest type of black hole."
        mock_generate_text.return_value = mock_response

        # 2. Ask a question about a topic that is not in the memory
        response, _, _ = self.pipeline.process_message("What is a supermassive black hole?")

        # 3. Assert that the mock was called, meaning the InquisitiveMind was activated
        mock_generate_text.assert_called_once()

        # 4. Assert that the response is the formatted output from the InquisitiveMind
        expected_response = f"I have learned that 'supermassive black hole' is: '{mock_response.strip()}'. Is this correct?"
        self.assertEqual(response, expected_response)


if __name__ == '__main__':
    unittest.main()
