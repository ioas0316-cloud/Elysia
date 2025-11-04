import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core_memory import Memory, EmotionalState
from Project_Sophia.gemini_api import APIKeyError, APIRequestError

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

    @patch('Project_Sophia.cognition_pipeline.LocalLLMCortex')
    @patch('Project_Sophia.cognition_pipeline.get_text_embedding')
    @patch('Project_Sophia.cognition_pipeline.generate_text')
    def test_conversational_memory_is_retrieved(self, mock_generate_text, mock_get_text_embedding, MockLocalLLMCortex):
        """
        Tests if the pipeline can retrieve a relevant past experience and use
        it in a response.
        """
        self.pipeline.api_available = True
        MockLocalLLMCortex.return_value.model = None
        mock_generate_text.return_value = "이전에 'I enjoy learning about black holes.'에 대해 이야기 나눈 것을 기억해요."
        mock_get_text_embedding.return_value = [0.1] * 768

        past_experience = Memory(
            timestamp="2025-01-01T12:00:00",
            content="I enjoy learning about black holes.",
            emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])
        )
        self.pipeline.core_memory.add_experience(past_experience)

        response, _ = self.pipeline.process_message("What do you know about black holes?")

        self.assertIn("black holes", response['text'])

    @patch('Project_Sophia.cognition_pipeline.LocalLLMCortex')
    @patch('Project_Sophia.journal_cortex.JournalCortex.write_journal_entry')
    @patch('Project_Sophia.cognition_pipeline.get_text_embedding')
    @patch('Project_Sophia.inquisitive_mind.generate_text')
    @patch('Project_Sophia.cognition_pipeline.generate_text')
    def test_inquisitive_mind_is_triggered(self, mock_cognition_generate_text, mock_inquisitive_generate_text, mock_get_text_embedding, mock_write_journal, MockLocalLLMCortex):
        """
        Tests if the InquisitiveMind is triggered when the pipeline encounters
        a question it cannot answer from memory or internal knowledge.
        """
        self.pipeline.api_available = True
        MockLocalLLMCortex.return_value.model = None
        mock_response = "A supermassive black hole is the largest type of black hole."
        mock_inquisitive_generate_text.return_value = mock_response
        mock_get_text_embedding.return_value = None
        mock_cognition_generate_text.return_value = "I don't know about 'supermassive black hole'. Seeking external knowledge."
        mock_write_journal.return_value = None

        self.pipeline.process_message("What is a supermassive black hole?")

        mock_inquisitive_generate_text.assert_called_once()

    @patch('Project_Sophia.cognition_pipeline.LocalLLMCortex')
    @patch('Project_Sophia.cognition_pipeline.generate_text', side_effect=APIKeyError("Test API Key Error"))
    def test_fallback_mechanism_on_api_key_error(self, mock_generate_text, MockLocalLLMCortex):
        """
        Tests that the pipeline's fallback mechanism is triggered on APIKeyError.
        """
        # Ensure the mock LLM instance reports no model, triggering the simple fallback
        MockLocalLLMCortex.return_value.model = None

        response, _ = self.pipeline.process_message("Tell me about photosynthesis.")
        self.assertIn("죄송합니다. 현재 주 지식망 및 보조 지식망에 모두 연결할 수 없습니다. 잠시 후 다시 시도해주세요.", response['text'])

    @patch('Project_Sophia.cognition_pipeline.LocalLLMCortex')
    @patch('Project_Sophia.cognition_pipeline.generate_text', side_effect=APIRequestError("Test API Request Error"))
    def test_fallback_mechanism_on_api_request_error(self, mock_generate_text, MockLocalLLMCortex):
        """
        Tests that the pipeline's fallback mechanism is triggered on APIRequestError.
        """
        MockLocalLLMCortex.return_value.model = None

        response, _ = self.pipeline.process_message("What is the weather like today?")
        self.assertIn("죄송합니다. 현재 주 지식망 및 보조 지식망에 모두 연결할 수 없습니다. 잠시 후 다시 시도해주세요.", response['text'])


    def test_purpose_oriented_workflow(self):
        """
        Tests the full purpose-oriented workflow, from '목적:' input to plan execution summary.
        """
        # 1. Define a purpose-oriented message
        purpose_message = "목적: 엘리시아의 자율 학습 능력을 강화하여 새로운 지식을 스스로 습득하게 한다."

        # 2. Process the message through the pipeline
        response, _ = self.pipeline.process_message(purpose_message)

        # 3. Assert the response is a summary of the executed plan
        self.assertIsInstance(response, dict)
        self.assertEqual(response.get('type'), 'text')
        response_text = response.get('text', '')

        self.assertIn("목적", response_text)
        self.assertIn("달성하기 위해 다음 계획을 실행했습니다", response_text)
        self.assertIn("실행 결과", response_text)
        self.assertIn("generate_conversational_response", response_text)


if __name__ == '__main__':
    unittest.main()
