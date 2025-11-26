import unittest
from Core.Mind.emotional_palette import EmotionalPalette

class TestEmotionalPalette(unittest.TestCase):
    def setUp(self):
        self.palette = EmotionalPalette()
        
    def test_mix_emotion(self):
        # Mix Joy and Sadness (Melancholy)
        components = {"Joy": 0.5, "Sadness": 0.5}
        qubit = self.palette.mix_emotion(components)
        
        # Check if properties are blended
        # Joy has high Alpha (0.8), Sadness low (0.2) -> Avg 0.5
        # After normalization (approx mag 0.73), alpha becomes ~0.685
        self.assertAlmostEqual(qubit.state.alpha.real, 0.685, delta=0.1)
        
        # Joy has high W (0.6), Sadness low (0.3) -> Avg 0.45
        self.assertAlmostEqual(qubit.state.w, 0.45, delta=0.1)
        
    def test_analyze_sentiment(self):
        text = "I am so happy but also a bit sad."
        scores = self.palette.analyze_sentiment(text)
        
        self.assertGreater(scores["Joy"], 0.0)
        self.assertGreater(scores["Sadness"], 0.0)
        self.assertEqual(scores["Anger"], 0.0)

if __name__ == '__main__':
    unittest.main()
