import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.playful_cortex import PlayfulCortex
from Project_Sophia.emotional_state import EmotionalState

class TestPlayfulCortex(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for each test."""
        self.mock_wave_mechanics = MagicMock()
        self.mock_sensory_cortex = MagicMock()
        self.cortex = PlayfulCortex(self.mock_wave_mechanics, self.mock_sensory_cortex)

    def test_play_generates_poetic_response(self):
        """
        Tests if the PlayfulCortex generates a creative, poetic response
        when engaged in play.
        """
        # 1. Configure mocks
        self.mock_wave_mechanics.kg_manager.get_random_node.return_value = {'id': '하늘'}
        self.mock_wave_mechanics.spread_activation.return_value = {
            '하늘': 1.0,
            '푸르름': 0.8,
            '자유': 0.6,
            '구름': 0.5
        }
        initial_emotion = EmotionalState(valence=0.1, arousal=0.1, dominance=0.0, primary_emotion="neutral", secondary_emotions=[])

        # 2. Engage in play
        response, new_emotion = self.cortex.play("오늘 날씨가 좋네.", initial_emotion)

        # 3. Assertions
        # Check that a poetic link was created
        self.assertTrue(any(phrase in response for phrase in [
            "떠오르는 건 왜일까요?",
            "보이지 않는 연결이 있는 것 같아요.",
            "피어나는 느낌이에요.",
            "그 멜로디일 거예요."
        ]))

        # Check that the concepts from wave mechanics are in the response
        self.assertIn("하늘", response)
        self.assertIn("푸르름", response)

        # Check for a positive emotional shift
        self.assertGreater(new_emotion.valence, initial_emotion.valence)
        self.assertGreater(new_emotion.arousal, initial_emotion.arousal)
        self.assertIn(new_emotion.primary_emotion, ["joy", "curiosity", "admiration"])

    def test_play_handles_no_associations(self):
        """
        Tests if the cortex provides a graceful fallback response when no
        strong associations are found.
        """
        # 1. Configure mocks to return no strong associations
        self.mock_wave_mechanics.kg_manager.get_random_node.return_value = {'id': '고독'}
        self.mock_wave_mechanics.spread_activation.return_value = {'고독': 1.0}
        initial_emotion = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, primary_emotion="neutral", secondary_emotions=[])

        # 2. Engage in play
        response, _ = self.cortex.play("...", initial_emotion)

        # 3. Assertions
        self.assertEqual(response, "'고독'(이)라는 단어를 생각하니, 왠지 마음이 평온해져요.")

if __name__ == '__main__':
    unittest.main()
