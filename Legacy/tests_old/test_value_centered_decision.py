import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow cross-project imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.value_centered_decision import ValueCenteredDecision
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.thought import Thought
from Project_Sophia.emotional_engine import EmotionalState
from tools.kg_manager import KGManager

class TestValueCenteredDecision(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for each test."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_kg_manager.kg = {
            'nodes': [
                {'id': 'love'}, {'id': 'hope'}, {'id': 'truth'},
                {'id': 'conflict'}, {'id': 'lies'}
            ]
        }

        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.vcd = ValueCenteredDecision(
            kg_manager=self.mock_kg_manager,
            wave_mechanics=self.mock_wave_mechanics,
            core_value='love'
        )

    def test_score_thought_integrates_multiple_factors(self):
        """
        Test that score_thought correctly combines resonance, confidence, and energy.
        """
        # This thought has high resonance and high confidence, but low energy.
        thought_A = Thought(
            content="희망은 사랑에서 비롯된다.",
            source='knowledge_graph',
            confidence=0.95,
            energy=5.0, # low energy
            evidence=[]
        )

        # This thought has lower resonance and confidence, but very high energy.
        thought_B = Thought(
            content="진실은 때로 갈등을 낳는다.",
            source='living_reason_system',
            confidence=0.7,
            energy=150.0, # high energy
            evidence=[]
        )

        # Mock the resonance calculation
        def mock_resonance_func(entity, core_value):
            if entity == 'hope': return 0.9
            if entity == 'truth': return 0.6
            if entity == 'conflict': return 0.2
            return 0.0
        self.mock_wave_mechanics.get_resonance_between.side_effect = mock_resonance_func

        # Mock the entity extractor
        def mock_extractor_func(text):
            if "희망" in text: return ['hope']
            if "진실" in text and "갈등" in text: return ['truth', 'conflict']
            return []

        with patch.object(self.vcd, '_find_mentioned_entities', side_effect=mock_extractor_func):
            score_A = self.vcd.score_thought(thought_A)
            score_B = self.vcd.score_thought(thought_B)

            # Check resonance calculation part
            # Resonance A (hope) = 0.9
            # Resonance B (truth, conflict) = (0.6 + 0.2) / sqrt(2) approx 0.56
            # VCD weights resonance heavily (1.5), so A should have a higher base score.
            #
            # Check energy calculation part
            # Energy A (5.0) -> log1p(5)/10 approx 0.17
            # Energy B (150.0) -> log1p(150)/10 approx 0.5
            # VCD weights energy (0.8), so B gets a significant energy bonus.

            # The final score depends on the weights, but we expect B's high energy
            # and A's high resonance/confidence to be the main drivers.
            # Let's verify the components are being called correctly.
            self.assertEqual(self.mock_wave_mechanics.get_resonance_between.call_count, 3)

            # Based on the weights (Res:1.5, Conf:1.0, Energy:0.8)
            # Score A ~ 1.5*0.9 + 1.0*0.95 + 0.8*0.17 = 1.35 + 0.95 + 0.136 = 2.436
            # Score B ~ 1.5*0.56 + 1.0*0.7 + 0.8*0.5 = 0.84 + 0.7 + 0.4 = 1.94
            # Therefore, thought A should be preferred.
            self.assertGreater(score_A, score_B)

    def test_suggest_thought_chooses_best_candidate(self):
        """
        Test that suggest_thought correctly chooses the best Thought object from a list.
        """
        thought_high_score = Thought(content="희망은 사랑의 증거이다.", source='kg', confidence=0.9)
        thought_low_score = Thought(content="거짓은 갈등을 만든다.", source='lrs', confidence=0.6, energy=20.0)

        candidates = [thought_low_score, thought_high_score]

        # Mock the scoring function to return predictable scores
        with patch.object(self.vcd, 'score_thought') as mock_score:
            def score_side_effect(candidate, *args, **kwargs):
                if "희망" in candidate.content:
                    return 3.0
                return 1.0
            mock_score.side_effect = score_side_effect

            best_thought = self.vcd.select_thought(candidates)

            # Verify the VCD chooses the thought with the higher mocked score.
            self.assertIs(best_thought, thought_high_score)
            self.assertEqual(mock_score.call_count, 2)

    def test_emotional_state_adjusts_thought_selection(self):
        """
        Tests that the primary emotion changes the weights and alters the outcome.
        """
        # A highly confident, static thought.
        thought_confident = Thought(content="태양은 빛난다.", source='kg', confidence=0.98, energy=1.0)
        # A less confident but highly energetic, dynamic thought.
        thought_energetic = Thought(content="사랑은 우주를 만든다.", source='lrs', confidence=0.6, energy=200.0)

        candidates = [thought_confident, thought_energetic]

        # Mock resonance and entity extraction to be neutral for this test
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5
        with patch.object(self.vcd, '_find_mentioned_entities', return_value=['태양', '사랑']):

            # Case 1: "calm" emotional state (should prefer confidence)
            calm_state = EmotionalState(0.5, 0.2, 0.5, primary_emotion='calm')
            best_thought_calm = self.vcd.select_thought(candidates, emotional_state=calm_state)
            self.assertIs(best_thought_calm, thought_confident)

            # Case 2: "joy" emotional state (should prefer energy)
            joy_state = EmotionalState(0.8, 0.7, 0.6, primary_emotion='joy')
            best_thought_joy = self.vcd.select_thought(candidates, emotional_state=joy_state)
            self.assertIs(best_thought_joy, thought_energetic)


    def test_wisdom_bonus_prefers_bone_source(self):
        """
        Tests that the 'Wisdom Bonus' correctly prefers thoughts from 'bone' (KG)
        over thoughts from 'flesh' (LRS) when other factors are equal.
        """
        # A thought from the foundational knowledge graph ('bone')
        thought_from_bone = Thought(
            content="사랑은 중요한 가치이다.",
            source='bone',
            confidence=0.9,
            energy=10.0
        )

        # An almost identical thought, but from the dynamic simulation ('flesh')
        thought_from_flesh = Thought(
            content="사랑은 중요한 가치이다.",
            source='flesh',
            confidence=0.9,
            energy=10.0
        )

        candidates = [thought_from_flesh, thought_from_bone]

        # Mock resonance and context to be identical for both thoughts
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.8
        with patch.object(self.vcd, '_find_mentioned_entities', return_value=['사랑']):

            # Use a neutral emotional state to not bias the weights
            neutral_state = EmotionalState(0.5, 0.5, 0.5, primary_emotion='neutral')

            # Score both thoughts
            score_bone = self.vcd.score_thought(thought_from_bone, emotional_state=neutral_state)
            score_flesh = self.vcd.score_thought(thought_from_flesh, emotional_state=neutral_state)

            # Assert that the score from 'bone' is significantly higher due to the bonus
            self.assertGreater(score_bone, score_flesh)
            self.assertAlmostEqual(score_bone, score_flesh + 0.5, delta=0.02) # Check if bonus is ~0.5

            # Verify that suggest_thought chooses the 'bone' thought
            best_thought = self.vcd.select_thought(candidates, emotional_state=neutral_state)
            self.assertIs(best_thought, thought_from_bone)


if __name__ == '__main__':
    unittest.main()
