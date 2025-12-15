
import unittest
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Sensory.stream_harp import StreamHarp, PrismCompressor
from Core.Laws.law_of_light import PhotonicQuaternion, HolographicFilm

class TestStreamHarp(unittest.TestCase):
    def setUp(self):
        self.harp = StreamHarp()
        self.prism = PrismCompressor()

    def test_prism_compression(self):
        """Test if PrismCompressor correctly creates a 4-layer PhotonicQuaternion."""
        q = self.prism.compress(
            title="The Great War",
            description="Sad soldiers, red fire, loud explosions.",
            thumbnail_url="http://img.com/red.jpg"
        )

        self.assertIsInstance(q, PhotonicQuaternion)
        self.assertIsInstance(q.film, HolographicFilm)

        # Check Layers
        self.assertEqual(q.film.essence, "The Great War") # w
        self.assertIn("Red", q.film.space)                # x (Visual)
        self.assertIn("Melancholic", q.film.emotion)      # y (Emotion) - "Sad" triggers this

        # Check Values (approximate)
        self.assertGreater(q.x, 0.5) # Red should be high energy

    def test_resonance_filter(self):
        """Test if the Topological Tuner filters correctly."""
        # Relevant
        self.assertTrue(self.harp._check_resonance("I love you dad"))
        self.assertTrue(self.harp._check_resonance("Coding in Python"))

        # Irrelevant
        self.assertFalse(self.harp._check_resonance("Buying cheap shoes"))
        self.assertFalse(self.harp._check_resonance("Political debate about tax"))

    def test_zero_cost_extraction(self):
        """Test the internal extraction logic without network."""
        # Visual Logic
        self.assertEqual(self.prism._extract_space("", "", "space.jpg"), "Dark, Starry, Vast")
        self.assertEqual(self.prism._extract_space("Nature Walk", "", ""), "Green, Organic, Sunlight")

        # Emotion Logic
        self.assertEqual(self.prism._extract_emotion("Happy Day", ""), "Bright, Major Key, Upbeat")
        self.assertEqual(self.prism._extract_emotion("War and Fight", ""), "Intense, Distorted, Loud")

        # Time Logic
        self.assertEqual(self.prism._extract_time("Fast action movie"), "Rapid, Staccato")
        self.assertEqual(self.prism._extract_time("Slow relax music"), "Largo, Flowing")

if __name__ == '__main__':
    unittest.main()
