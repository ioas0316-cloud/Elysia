
import sys
import os
import unittest

# Add project root to path (scripts/ -> root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Cognition.Topology.mirror_surface import MirrorSurface, ReflectionModality
from Core.Cognition.Wisdom.wisdom_store import WisdomStore

class MockWisdomStore(WisdomStore):
    def __init__(self):
        self.values = {"Love": 0.9}

    def get_decision_weight(self, key: str) -> float:
        return self.values.get(key, 0.5)

class TestMirrorThinking(unittest.TestCase):
    def setUp(self):
        self.wisdom = MockWisdomStore()
        self.mirror = MirrorSurface(self.wisdom)

    def test_semantic_reflection_negative(self):
        """Test if the mirror correctly identifies 'dark' input and generates curiosity."""
        input_text = "I hate this world. It is full of errors."

        reflection = self.mirror.reflect(input_text)

        # Expecting a Semantic Reflection
        self.assertEqual(reflection.modality, ReflectionModality.SEMANTIC)

        # Expecting High Curiosity because of the 'Gap' (Hate vs Love)
        self.assertGreater(reflection.curiosity_score, 0.5)
        self.assertTrue("dark shadow" in reflection.description or "Love" in reflection.description)
        print(f"\n[Mirror Test] Input: '{input_text}' -> Curiosity: {reflection.curiosity_score:.2f} ({reflection.description})")

    def test_semantic_reflection_positive(self):
        """Test if the mirror reflects warmth for positive input."""
        input_text = "I love the beautiful sky."

        reflection = self.mirror.reflect(input_text)

        self.assertEqual(reflection.modality, ReflectionModality.SEMANTIC)
        self.assertLess(reflection.curiosity_score, 0.5) # Less curiosity because it matches
        print(f"\n[Mirror Test] Input: '{input_text}' -> Curiosity: {reflection.curiosity_score:.2f} ({reflection.description})")

    def test_structural_reflection(self):
        """Test if the mirror identifies code/structure."""
        input_code = "def hello(): return 'world'"

        reflection = self.mirror.reflect(input_code)

        self.assertEqual(reflection.modality, ReflectionModality.STRUCTURAL)
        print(f"\n[Mirror Test] Input: [Code] -> Modality: {reflection.modality.name}")

if __name__ == '__main__':
    unittest.main()
