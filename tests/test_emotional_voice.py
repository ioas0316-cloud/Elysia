import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.emotional_engine import EmotionalState

class TestEmotionalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline and mock its dependencies for each test."""
        self.pipeline = CognitionPipeline()
        self.pipeline.reasoner = MagicMock()
        self.pipeline.vcd = MagicMock()
        self.pipeline.insight_synthesizer = MagicMock()

    def test_joyful_response(self):
        """Tests that a 'joy' emotional state styles the response positively."""
        self.pipeline.current_emotional_state = EmotionalState(valence=0.8, arousal=0.6, dominance=0.3, primary_emotion='joy')
        fact = "'소크라테스'은(는) '인간'의 한 종류예요."
        self.pipeline.reasoner.deduce_facts.return_value = [fact]
        self.pipeline.vcd.suggest_action.return_value = fact
        self.pipeline.insight_synthesizer.synthesize.return_value = fact

        response, _ = self.pipeline.process_message("소크라테스에 대해 알려줘")

        # Updated expected response to match the actual output of ResponseStyler
        expected_response = f"정말 기뻐요! {fact} 🎉"
        self.assertEqual(response['text'], expected_response)

    def test_sad_response(self):
        """Tests that a 'sadness' emotional state styles the response with a somber tone."""
        self.pipeline.current_emotional_state = EmotionalState(valence=-0.7, arousal=-0.5, dominance=-0.2, primary_emotion='sadness')
        fact = "'플루토'은(는) 더 이상 '행성'이 아니에요."
        self.pipeline.reasoner.deduce_facts.return_value = [fact]
        self.pipeline.vcd.suggest_action.return_value = fact
        self.pipeline.insight_synthesizer.synthesize.return_value = fact

        response, _ = self.pipeline.process_message("플루토에 대해 알려줘")

        # Updated expected response
        expected_response = f"조금 슬픈 마음이 들지만... {fact} 😔"
        self.assertEqual(response['text'], expected_response)

    def test_neutral_response(self):
        """Tests that a neutral emotional state does not add extra styling."""
        self.pipeline.current_emotional_state = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion='neutral')
        fact = "'물'은 'H2O'로 구성되어 있어요."
        self.pipeline.reasoner.deduce_facts.return_value = [fact]
        self.pipeline.vcd.suggest_action.return_value = fact
        self.pipeline.insight_synthesizer.synthesize.return_value = fact

        response, _ = self.pipeline.process_message("물에 대해 알려줘")

        expected_response = f"저는 이렇게 생각해요: {fact}"
        self.assertEqual(response['text'], expected_response)

if __name__ == '__main__':
    unittest.main()
