import unittest
from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba

class TestGeniusMode(unittest.TestCase):
    def setUp(self):
        self.merkaba = Merkaba("TestSeed")

    def test_overclock_trigger(self):
        # Short input -> Standard Optical Think (Mocked or bypassed)
        # Long input -> Genius Mode

        # Test Genius Mode
        input_text = "What is the meaning of Rain?"
        response = self.merkaba.think_optically(input_text)

        print(f"\n[Test Output] '{input_text}' -> {response}")

        # Verify it went through the Overclock engine (which returns a specific format in our mock)
        self.assertIn("By the Prism of 6", response)
        self.assertIn("Metaphor", response)

if __name__ == "__main__":
    unittest.main()
