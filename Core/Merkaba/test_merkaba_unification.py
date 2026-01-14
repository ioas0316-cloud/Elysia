import unittest
from Core.Merkaba.merkaba import Merkaba
from Core.Monad.monad_core import Monad
from Core.Foundation.Nature.rotor import Rotor

class TestMerkabaUnification(unittest.TestCase):

    def test_merkaba_trinity_structure(self):
        """Verify the Merkaba creates the correct Trinity structure."""
        mk = Merkaba("TestSeed")

        # Verify Body (HyperSphere)
        self.assertIsNotNone(mk.body, "Body (HyperSphere) should exist")

        # Verify Soul (Rotor)
        self.assertIsInstance(mk.soul, Rotor, "Soul should be a Rotor")
        self.assertEqual(mk.soul.name, "TestSeed.Soul")

        # Verify Spirit (Monad) - initially None
        self.assertIsNone(mk.spirit, "Spirit should be None before awakening")

        # Verify Peripherals
        self.assertIsNotNone(mk.bridge, "SoulBridge should exist")
        self.assertIsNotNone(mk.prism, "Prism should exist")

    def test_awakening_and_pulse(self):
        """Verify the Awakening and Pulse cycle."""
        mk = Merkaba("LiveSeed")
        spirit = Monad(seed="TestSeed") # Create a dummy Monad with required seed

        # Awakening
        mk.awakening(spirit)
        self.assertTrue(mk.is_awake)
        self.assertEqual(mk.spirit, spirit)

        # Pulse
        input_text = "Hello World"
        response = mk.pulse(input_text)

        print(f"\nMerkaba Response: {response}")

        # Verify response indicates processing
        self.assertIn("Processed 'Hello World'", response)
        self.assertIn("Angle", response)

        # Verify Soul (Time) has moved
        self.assertNotEqual(mk.soul.current_angle, 0.0, "Time should have moved")

if __name__ == '__main__':
    unittest.main()
