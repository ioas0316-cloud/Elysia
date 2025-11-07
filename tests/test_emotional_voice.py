import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.emotional_engine import EmotionalState
from tools.kg_manager import KGManager

class TestEmotionalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline and a mock KG for each test."""
        self.pipeline = CognitionPipeline()

        # Mock the method that overwrites the emotional state during tests.
        # This allows us to manually set the state and have it persist.
        self.pipeline._update_emotional_state = MagicMock()

        # Mock the entire LogicalReasoner to prevent it from calling internal methods
        # on a mock KGManager, which was causing errors.
        self.mock_reasoner = MagicMock()
        self.pipeline.reasoner = self.mock_reasoner

        # Set api_available to False to ensure we are testing the internal response
        self.pipeline.api_available = False

    def test_joyful_response(self):
        """Tests that a 'joy' emotional state styles the response positively."""
        # Arrange: Set the emotional state to 'joy'
        self.pipeline.current_emotional_state = EmotionalState(valence=0.8, arousal=0.6, dominance=0.3, primary_emotion='joy')

        # Arrange: Mock the reasoner to return a specific fact
        fact = "'소크라테스'은(는) '인간'의 한 종류예요."
        self.mock_reasoner.deduce_facts.return_value = [fact]

        # Act
        response, _ = self.pipeline.process_message("소크라테스에 대해 알려줘")

        # Assert
        prefix = "나는 지금 네 뜻을 더 선명히 이해하고자 해."
        expected_body = f"와, 좋은데요! {fact}!"
        expected_response = f"{prefix} {expected_body}"
        self.assertEqual(response['text'], expected_response)

    def test_sad_response(self):
        """Tests that a 'sadness' emotional state styles the response with a somber tone."""
        # Arrange: Set the emotional state to 'sadness'
        self.pipeline.current_emotional_state = EmotionalState(valence=-0.7, arousal=-0.5, dominance=-0.2, primary_emotion='sadness')

        # Arrange: Mock the reasoner to return a specific fact
        fact = "'플루토'은(는) 더 이상 '행성'이 아니에요."
        self.mock_reasoner.deduce_facts.return_value = [fact]

        # Act
        response, _ = self.pipeline.process_message("플루토에 대해 알려줘")

        # Assert
        prefix = "나는 지금 네 뜻을 더 선명히 이해하고자 해."
        expected_body = f"조금 슬픈 이야기지만... {fact}"
        expected_response = f"{prefix} {expected_body}"
        self.assertEqual(response['text'], expected_response)

    def test_neutral_response(self):
        """Tests that a neutral emotional state does not add extra styling."""
        # Arrange: Set a neutral emotional state
        self.pipeline.current_emotional_state = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion='neutral')

        # Arrange: Mock the reasoner to return a specific fact
        fact = "'물'은 'H2O'로 구성되어 있어요."
        self.mock_reasoner.deduce_facts.return_value = [fact]

        # Act
        response, _ = self.pipeline.process_message("물에 대해 알려줘")

        # Assert
        prefix = "나는 지금 네 뜻을 더 선명히 이해하고자 해."
        expected_response = f"{prefix} {fact}"
        self.assertEqual(response['text'], expected_response)

if __name__ == '__main__':
    unittest.main()
