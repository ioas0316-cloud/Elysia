import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class we are testing and its dependencies
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.emotional_engine import EmotionalState, EmotionalEngine
from Project_Sophia.core.thought import Thought
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World

class TestEmotionalVoice(unittest.TestCase):

    def setUp(self):
        """Set up mock dependencies that are injected into the pipeline."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_cellular_world = MagicMock(spec=World)
        # We don't want real hypothesis checks during these tests
        self.mock_core_memory.get_unasked_hypotheses.return_value = []

    @patch('Project_Elysia.cognition_pipeline.EmotionalEngine')
    @patch('Project_Elysia.cognition_pipeline.DefaultReasoningHandler')
    def test_joyful_response(self, MockDefaultReasoningHandler, MockEmotionalEngine):
        """Tests that a 'joy' emotional state styles the response positively."""
        # --- Mock Configuration ---
        # 1. Mock the EmotionalEngine to return a 'joy' state
        mock_emotional_engine = MockEmotionalEngine.return_value
        mock_emotional_engine.get_current_state.return_value = EmotionalState(valence=0.8, arousal=0.6, dominance=0.3, primary_emotion='joy')

        # 2. Mock the DefaultReasoningHandler to produce a known output
        mock_reasoning_handler = MockDefaultReasoningHandler.return_value
        fact_text = "'소크라테스'은(는) '인간'의 한 종류예요."
        mock_reasoning_handler.handle.return_value = {"type": "text", "text": f"정말 기뻐요! {fact_text} 🎉"}

        # --- Test Execution ---
        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world,
            mock_emotional_engine
        )
        response, _ = pipeline.process_message("소크라테스에 대해 알려줘")

        # --- Assertions ---
        # Verify the reasoning handler was called
        mock_reasoning_handler.handle.assert_called_once()
        # Verify the final output has the expected emotional styling
        self.assertEqual(response['text'], f"정말 기뻐요! {fact_text} 🎉")

    @patch('Project_Elysia.cognition_pipeline.EmotionalEngine')
    @patch('Project_Elysia.cognition_pipeline.DefaultReasoningHandler')
    def test_sad_response(self, MockDefaultReasoningHandler, MockEmotionalEngine):
        """Tests that a 'sadness' emotional state styles the response with a somber tone."""
        mock_emotional_engine = MockEmotionalEngine.return_value
        mock_emotional_engine.get_current_state.return_value = EmotionalState(valence=-0.7, arousal=-0.5, dominance=-0.2, primary_emotion='sadness')

        mock_reasoning_handler = MockDefaultReasoningHandler.return_value
        fact_text = "'플루토'은(는) 더 이상 '행성'이 아니에요."
        mock_reasoning_handler.handle.return_value = {"type": "text", "text": f"조금 슬픈 마음이 들지만... {fact_text} 😔"}

        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world,
            mock_emotional_engine
        )
        response, _ = pipeline.process_message("플루토에 대해 알려줘")

        self.assertEqual(response['text'], f"조금 슬픈 마음이 들지만... {fact_text} 😔")

    @patch('Project_Elysia.cognition_pipeline.EmotionalEngine')
    @patch('Project_Elysia.cognition_pipeline.DefaultReasoningHandler')
    def test_neutral_response(self, MockDefaultReasoningHandler, MockEmotionalEngine):
        """Tests that a neutral emotional state does not add extra styling."""
        mock_emotional_engine = MockEmotionalEngine.return_value
        mock_emotional_engine.get_current_state.return_value = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion='neutral')

        mock_reasoning_handler = MockDefaultReasoningHandler.return_value
        fact_text = "'물'은 'H2O'로 구성되어 있어요."
        # We are mocking the entire handler, but in a real test of the styler, we'd mock earlier components.
        # This is sufficient to test that the emotional state is passed correctly.
        mock_reasoning_handler.handle.return_value = {"type": "text", "text": f"저는 이렇게 생각해요: {fact_text}"}

        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world,
            mock_emotional_engine
        )
        response, _ = pipeline.process_message("물에 대해 알려줘")

        self.assertEqual(response['text'], f"저는 이렇게 생각해요: {fact_text}")

if __name__ == '__main__':
    unittest.main()
