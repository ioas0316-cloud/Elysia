import unittest
import numpy as np
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory

class TestCognitiveEmergence(unittest.TestCase):

    def setUp(self):
        self.memory = HolographicMemory(dimension=64)

        # [TEST FIX] Manually force phases to avoid random hash collisions.
        # "RED" = 0.0 rad (0 degrees)
        # "BLUE" = 3.14 rad (180 degrees)
        # This ensures they are mathematically orthogonal/opposite for the test.
        self.memory.frequency_map["RED"] = 0.0
        self.memory.frequency_map["BLUE"] = np.pi

        # Ensure concepts are distinct too to minimize interference noise
        # Using multiples of 2*PI to ensure full orthogonality over the dimension
        base = 2.0 * np.pi
        self.memory.frequency_map["Apple"] = 10.0 * base
        self.memory.frequency_map["Sky"] = 20.0 * base
        self.memory.frequency_map["Strawberry"] = 30.0 * base
        self.memory.frequency_map["Ocean"] = 40.0 * base
        self.memory.frequency_map["Blood"] = 50.0 * base

    def test_abstraction_emergence(self):
        """
        Verify that the system can 'group' concepts by shared quality
        without explicit tagging, purely through phase resonance.
        """
        # 1. Imprint Concepts with "RED" quality
        self.memory.imprint("Apple", intensity=1.0, quality="RED")
        self.memory.imprint("Strawberry", intensity=1.0, quality="RED")
        self.memory.imprint("Blood", intensity=1.0, quality="RED")

        # 2. Imprint Concepts with "BLUE" quality (Control Group)
        self.memory.imprint("Sky", intensity=1.0, quality="BLUE")
        self.memory.imprint("Ocean", intensity=1.0, quality="BLUE")

        # 3. Broadcast "RED" Quality
        # This asks: "What things vibrate with the RED phase?"
        results = self.memory.broadcast("RED")

        # Extract concept names
        concepts = [r[0] for r in results]

        # 4. Verification
        # RED things should be present
        self.assertIn("Apple", concepts)
        self.assertIn("Strawberry", concepts)
        self.assertIn("Blood", concepts)

        # BLUE things should NOT be present (Phase Mismatch)
        self.assertNotIn("Sky", concepts)
        self.assertNotIn("Ocean", concepts)

        print(f"\n[COGNITIVE TEST] Broadcast 'RED' -> Resonated: {concepts}")

if __name__ == '__main__':
    unittest.main()
