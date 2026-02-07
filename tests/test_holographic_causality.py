import unittest
import numpy as np
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory
from Core.S1_Body.L4_Causality.causal_flow_engine import CausalFlowEngine

class TestHolographicCausality(unittest.TestCase):

    def setUp(self):
        # Initialize a small manifold for testing
        self.memory = HolographicMemory(dimension=32)
        self.engine = CausalFlowEngine(self.memory)

    def test_imprint_and_resonate(self):
        """
        Verify that imprinting a concept allows it to resonate later.
        """
        concept = "Apple"
        self.memory.imprint(concept, intensity=1.0)

        # Check Resonance
        # We manually call resonate for testing the memory component
        (res_concept, amplitude, phase) = self.memory.resonate(concept)

        # Amplitude should be significant (near 1.0) for a known concept
        self.assertGreater(amplitude, 0.9)
        self.assertEqual(res_concept, "Apple")

    def test_dissonance_check(self):
        """
        Verify that conflicting qualities create dissonance (low energy match).
        """
        # Imprint "Apple" with "RED" quality
        self.memory.imprint("Apple", intensity=1.0, quality="RED")

        # Check against "Apple" with "BLUE" quality
        # This should have lower resonance/higher dissonance
        dissonance = self.memory.check_dissonance("Apple", "BLUE")

        # Expect high dissonance because "RED" != "BLUE"
        # Note: In our simple prototype, different qualities shift phases differently
        # so the dot product will be smaller than 1.0.
        self.assertGreater(dissonance, 0.1)

    def test_causal_flow_cycle(self):
        """
        Verify the full Ignition -> Resonance -> Collapse cycle.
        """
        # 1. Ignite a new concept
        intent = "NewIdea"
        packet = self.engine.ignite(intent)
        self.assertEqual(packet["seed"], "NewIdea")

        # 2. Flow (Resonance Check) - Should be DISSONANCE (New)
        flow_state = self.engine.flow(packet)
        self.assertEqual(flow_state["flow_type"], "DISSONANCE")

        # 3. Collapse (Result) - Should trigger GENESIS
        result = self.engine.collapse(flow_state)
        self.assertIn("[GENESIS]", result)

        # Now, actually Imprint it to simulate learning
        self.memory.imprint("NewIdea", intensity=1.0)

        # 4. Re-run Flow - Should be HARMONY
        flow_state_2 = self.engine.flow(packet)
        self.assertEqual(flow_state_2["flow_type"], "HARMONY")

        # 5. Re-run Collapse - Should trigger MANIFEST
        result_2 = self.engine.collapse(flow_state_2)
        self.assertIn("[MANIFEST]", result_2)

if __name__ == '__main__':
    unittest.main()
