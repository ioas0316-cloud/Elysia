"""
[PHASE 28: UNITY TESTAMENT]
Test Merkaba Unification: Validating the convergence of the Chariot and the Pilot.
"A test is not a check for failure, but a celebration of alignment."
"""
import unittest
from Core.S1_Body.L6_Structure.M1_Merkaba.merkaba import Merkaba
from Core.S1_Body.L7_Spirit.M1_Monad.monad_core import Monad
from Core.S1_Body.L6_Structure.Nature.rotor import Rotor, RotorMask

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
        """Verify the Awakening and Pulse cycle with Bitmask."""
        mk = Merkaba("LiveSeed")
        try:
            spirit = Monad(seed="TestSeed")
        except NameError:
            # Fallback if Monad isn't imported (though in this test file it is hard imported)
            # This is just a safeguard structure if we were to mock it, but for now we proceed.
            # Real fix: Create a MockMonad if Monad is missing.
            class MockMonad:
                 def __init__(self, seed): pass
            spirit = MockMonad(seed="TestSeed")

        mk.awakening(spirit)

        # Test POINT Mode (Fact)
        # Should process as a single snapshot
        response_point = mk.pulse("What is truth?", mode="POINT")
        print(f"\nPOINT Response: {response_point}")
        self.assertIn("Mode: POINT", response_point)
        self.assertIn("Items: 1", response_point)

        # Test LINE Mode (Flow)
        # Should process as a stream (e.g., 3 items)
        response_line = mk.pulse("Tell me a story", mode="LINE")
        print(f"\nLINE Response: {response_line}")
        self.assertIn("Mode: LINE", response_line)
        self.assertIn("Items: 3", response_line)

if __name__ == '__main__':
    unittest.main()
