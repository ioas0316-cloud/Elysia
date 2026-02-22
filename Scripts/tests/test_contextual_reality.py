import unittest
import numpy as np
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory
from Core.S1_Body.L4_Causality.causal_flow_engine import CausalFlowEngine

class TestContextualReality(unittest.TestCase):

    def setUp(self):
        self.memory = HolographicMemory(dimension=64)
        self.engine = CausalFlowEngine(self.memory)

        # [SETUP] Orthogonal Frequencies
        base = 2.0 * np.pi
        self.memory.frequency_map["Sky"] = 10.0 * base
        self.memory.frequency_map["BLUE"] = 20.0 * base
        self.memory.frequency_map["RED"] = 30.0 * base

    def test_contextual_acceptance(self):
        """
        Verify that the system records the trajectory of thought.
        """
        # 1. Imprint Normal Reality (Sky is Blue)
        self.memory.imprint("Sky", intensity=1.0, quality="BLUE")

        # 2. Ignite with "Sky"
        # We manually inject RED quality (Dissonance) into the query for simulation
        # In a real engine, the input would be "Red Sky".
        # Here we just check if the engine logs the flow.

        packet = self.engine.ignite("Sky")

        # 3. Flow (Rotor Spins)
        flow_state = self.engine.flow(packet)

        # 4. Collapse (Spirit Judges)
        result = self.engine.collapse(flow_state)

        print(f"\n[CONTEXT TEST] Result: {result}")

        # 5. Verify Trajectory Log
        # The result string should contain the path narrative
        self.assertIn("| Path:", result)
        self.assertIn("Truth", result) # Should be HARMONY since we queried Sky against Sky

if __name__ == '__main__':
    unittest.main()
