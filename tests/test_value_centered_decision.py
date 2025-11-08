import unittest
from unittest.mock import MagicMock, patch

# Modules to be tested
from Project_Elysia.value_centered_decision import VCD
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestValueCenteredDecision(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for each test."""
        # Mock the KGManager to simulate finding entities
        self.mock_kg_manager = MagicMock(spec=KGManager)
        # Mock the knowledge graph nodes
        self.mock_kg_manager.kg = {
            'nodes': [
                {'id': 'love'},
                {'id': 'sacrifice'},
                {'id': 'hatred'}
            ]
        }

        # Mock the WaveMechanics to return predefined resonance scores
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)

        # The VCD instance to be tested, with mocked dependencies
        self.vcd = VCD(
            kg_manager=self.mock_kg_manager,
            wave_mechanics=self.mock_wave_mechanics,
            core_value='love'
        )

    def test_value_alignment_with_high_resonance_concept(self):
        """
        Test that a concept with high resonance to 'love' gets a high alignment score.
        """
        # Configure the mock to return a high resonance for 'sacrifice' -> 'love'
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.8

        # The VCD's entity extractor should find 'sacrifice' in the text
        with patch.object(self.vcd, '_extract_entities_from_text', return_value=['sacrifice']) as mock_extractor:
            score = self.vcd.value_alignment("희생은 숭고한 가치이다.")

            # Assert that the extractor was called
            mock_extractor.assert_called_once_with("희생은 숭고한 가치이다.")
            # Assert that wave_mechanics was called with the correct arguments
            self.mock_wave_mechanics.get_resonance_between.assert_called_once_with('sacrifice', 'love')
            # Assert that the final score is high (should be 0.8 clamped to 1.0)
            self.assertAlmostEqual(score, 0.8)

    def test_value_alignment_with_low_resonance_concept(self):
        """
        Test that a concept with low resonance to 'love' gets a low alignment score.
        """
        # Configure the mock to return a low resonance for 'hatred' -> 'love'
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.1

        with patch.object(self.vcd, '_extract_entities_from_text', return_value=['hatred']) as mock_extractor:
            score = self.vcd.value_alignment("증오는 모든 것을 파괴한다.")

            mock_extractor.assert_called_once_with("증오는 모든 것을 파괴한다.")
            self.mock_wave_mechanics.get_resonance_between.assert_called_once_with('hatred', 'love')
            self.assertAlmostEqual(score, 0.1)

    def test_suggest_action_chooses_higher_resonance_candidate(self):
        """
        Test that suggest_action correctly chooses the candidate with higher value alignment.
        """
        # Define the resonance scores for our test concepts
        def mock_resonance_func(entity, core_value):
            if entity == 'sacrifice':
                return 0.9
            if entity == 'hatred':
                return 0.15
            return 0.0

        self.mock_wave_mechanics.get_resonance_between.side_effect = mock_resonance_func

        # Two candidate sentences to be evaluated
        candidates = [
            "증오는 강력한 힘이다.", # Should have low score
            "희생은 사랑의 증거이다."  # Should have high score
        ]

        # Mock the entity extractor for both sentences
        def mock_extractor_func(text):
            if "희생" in text:
                return ['sacrifice']
            if "증오" in text:
                return ['hatred']
            return []

        with patch.object(self.vcd, '_extract_entities_from_text', side_effect=mock_extractor_func):
            # We are ignoring context_fit and freshness for this test by mocking them
            with patch.object(self.vcd, 'context_fit', return_value=0.5):
                with patch.object(self.vcd, 'freshness', return_value=1.0):
                    best_action = self.vcd.suggest_action(candidates)

                    # The VCD should choose the sentence about 'sacrifice'
                    self.assertEqual(best_action, "희생은 사랑의 증거이다.")

if __name__ == '__main__':
    unittest.main()
