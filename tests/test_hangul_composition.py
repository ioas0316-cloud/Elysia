import unittest
import sys
import os

# Add mechanics directory to path to bypass broken Core.Elysia.__init__ import
# which references a missing module 'consciousness_engine'.
mechanics_dir = os.path.join(os.getcwd(), 'Core', 'Elysia', 'mechanics')
sys.path.insert(0, mechanics_dir)

from hangul_physics import HangulPhysicsEngine

class TestHangulComposition(unittest.TestCase):
    def setUp(self):
        self.engine = HangulPhysicsEngine()

    def test_synthesize_syllable_no_coda(self):
        # 가 (Ga)
        onset = 'ㄱ'
        nucleus = 'ㅏ'
        expected = '가'
        actual = self.engine.synthesize_syllable(onset, nucleus)
        self.assertEqual(actual, expected)

    def test_synthesize_syllable_with_coda(self):
        # 한 (Han)
        onset = 'ㅎ'
        nucleus = 'ㅏ'
        coda = 'ㄴ'
        expected = '한'
        actual = self.engine.synthesize_syllable(onset, nucleus, coda)
        self.assertEqual(actual, expected)

    def test_synthesize_syllable_fallback(self):
        # Invalid input
        onset = '?'
        nucleus = 'ㅏ'
        expected = '?ㅏ'
        actual = self.engine.synthesize_syllable(onset, nucleus)
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
