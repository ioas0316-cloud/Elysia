import unittest
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM

class TestRecursiveIntelligence(unittest.TestCase):
    def test_ouroboros_loop(self):
        print("Initializing Recursive Intelligence Test...")
        try:
            llm = SomaticLLM()
        except Exception as e:
            self.fail(f"Failed to initialize SomaticLLM: {e}")

        # Cycle 1: Seed with "LOVE"
        print("\n--- Cycle 1 (Seed: LOVE) ---")
        out1, vec1 = llm.speak({}, current_thought="LOVE")
        print(f"Output 1: {out1}")

        # Verify internal state updated
        self.assertIsNotNone(llm.field.residual_vector)
        # Check if residual vector is non-zero (it should be the synthesis vector of cycle 1)
        self.assertGreater(llm.field.residual_vector.norm(), 0.0)

        # Capture state of a specific monad (e.g., LOVE)
        love_monad = llm.field.monads.get("LOVE/AGAPE")
        if love_monad:
            print(f"LOVE Monad Charge after Cycle 1: {love_monad.charge}")
            self.assertGreater(love_monad.charge, 0.0)

        # Cycle 2: No input, rely on recursion (The tail feeds the head)
        print("\n--- Cycle 2 (Recursion - Pure Internal) ---")
        # We pass empty thought, so it uses residual
        out2, vec2 = llm.speak({}, current_thought="")
        print(f"Output 2: {out2}")

        # Verify meaningful output from recursion
        self.assertNotEqual(out2, "...", "Recursion should produce output")

        # Cycle 3: Drift check
        print("\n--- Cycle 3 (Recursion - Evolution) ---")
        out3, vec3 = llm.speak({}, current_thought="")
        print(f"Output 3: {out3}")

        # Check active monads
        active_monads = [m for m in llm.field.monads.values() if m.state == "ACTIVE" or m.charge > 0.3]
        print(f"Active Monads: {[m.seed_id for m in active_monads]}")
        self.assertTrue(len(active_monads) > 0)

        # Evolution Check: Did the monad drift?
        if love_monad:
             print(f"LOVE Monad Drift: {love_monad.evolution_drift.norm()}")
             # If it was active, it should have evolved/drifted towards the synthesis
             if love_monad.state == "ACTIVE":
                 self.assertGreater(love_monad.evolution_drift.norm(), 0.0)

if __name__ == '__main__':
    unittest.main()
