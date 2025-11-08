import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.emotional_engine import EmotionalState
from Project_Sophia.core.thought import Thought

class TestEmotionalVoice(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline and mock its dependencies for each test."""
        self.pipeline = CognitionPipeline()
        self.pipeline.reasoner = MagicMock()
        self.pipeline.vcd = MagicMock()
        self.pipeline.insight_synthesizer = MagicMock()
        self.pipeline._check_and_verify_hypotheses = MagicMock(return_value=None)

    def test_joyful_response(self):
        """Tests that a 'joy' emotional state styles the response positively."""
        self.pipeline.current_emotional_state = EmotionalState(valence=0.8, arousal=0.6, dominance=0.3, primary_emotion='joy')
        fact_text = "'소크라테스'은(는) '인간'의 한 종류예요."
        thought = Thought(content=fact_text, source="KG", confidence=0.9, energy=0.8)
        self.pipeline.reasoner.deduce_facts.return_value = [thought]
        self.pipeline.vcd.suggest_thought.return_value = thought
        self.pipeline.vcd.score_thought.return_value = 1.0  # Mock the score to ensure logical path
        self.pipeline.insight_synthesizer.synthesize.return_value = fact_text

        response, _ = self.pipeline.process_message("소크라테스에 대해 알려줘")

        expected_response = f"정말 기뻐요! {fact_text} 🎉"
        self.assertEqual(response['text'], expected_response)

    def test_sad_response(self):
        """Tests that a 'sadness' emotional state styles the response with a somber tone."""
        self.pipeline.current_emotional_state = EmotionalState(valence=-0.7, arousal=-0.5, dominance=-0.2, primary_emotion='sadness')
        fact_text = "'플루토'은(는) 더 이상 '행성'이 아니에요."
        thought = Thought(content=fact_text, source="KG", confidence=0.9, energy=0.2)
        self.pipeline.reasoner.deduce_facts.return_value = [thought]
        self.pipeline.vcd.suggest_thought.return_value = thought
        self.pipeline.vcd.score_thought.return_value = 1.0
        self.pipeline.insight_synthesizer.synthesize.return_value = fact_text

        response, _ = self.pipeline.process_message("플루토에 대해 알려줘")

        expected_response = f"조금 슬픈 마음이 들지만... {fact_text} 😔"
        self.assertEqual(response['text'], expected_response)

    def test_neutral_response(self):
        """Tests that a neutral emotional state does not add extra styling."""
        self.pipeline.current_emotional_state = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion='neutral')
        fact_text = "'물'은 'H2O'로 구성되어 있어요."
        thought = Thought(content=fact_text, source="KG", confidence=0.99, energy=0.5)
        self.pipeline.reasoner.deduce_facts.return_value = [thought]
        self.pipeline.vcd.suggest_thought.return_value = thought
        self.pipeline.vcd.score_thought.return_value = 1.0
        self.pipeline.insight_synthesizer.synthesize.return_value = fact_text

        response, _ = self.pipeline.process_message("물에 대해 알려줘")

        expected_response = f"저는 이렇게 생각해요: {fact_text}"
        self.assertEqual(response['text'], expected_response)

if __name__ == '__main__':
    unittest.main()
