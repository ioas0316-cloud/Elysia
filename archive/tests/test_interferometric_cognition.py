import sys
import os
import math
import unittest

# Add root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import SovereignVector, SovereignInterferometer

class TestInterferometricCognition(unittest.TestCase):
    def test_interferometer_dynamic_binary(self):
        inter = SovereignInterferometer()

        # 1. Create a Sovereign Reference '1'
        ref_vec = SovereignVector([1.0] * 21).normalize()
        inter.set_sovereign_reference(ref_vec, label="Unity")

        # 2. Test 'Same' Signal (Resonance ~ 1.0, State = 1)
        signal_same = SovereignVector([0.95] * 21).normalize()
        diff_same = inter.perceive_difference(signal_same)

        print(f"\n[SAME] Res: {diff_same['resonance']:.3f}, ΔΦ: {diff_same['delta_phi']:.3f}, State: {diff_same['state']}")
        self.assertGreater(diff_same['resonance'], 0.9)
        self.assertEqual(diff_same['state'], 1)

        # 3. Test 'Different' Signal (Resonance ~ 0.0, State = 0)
        # Create an orthogonal vector
        data_diff = [0.0] * 21
        data_diff[0] = 1.0
        data_diff[1] = -1.0 # Simple way to get low resonance against a vector of all 1s
        signal_diff = SovereignVector(data_diff).normalize()

        diff_other = inter.perceive_difference(signal_diff)
        print(f"[DIFF] Res: {diff_other['resonance']:.3f}, ΔΦ: {diff_other['delta_phi']:.3f}, State: {diff_other['state']}")
        self.assertLess(diff_other['resonance'], 0.1)
        self.assertEqual(diff_other['state'], 0)

        # 4. Test 'Interference' (Middle State)
        data_inter = [1.0] * 10 + [0.0] * 11
        signal_inter = SovereignVector(data_inter).normalize()

        diff_inter = inter.perceive_difference(signal_inter)
        print(f"[INTER] Res: {diff_inter['resonance']:.3f}, ΔΦ: {diff_inter['delta_phi']:.3f}, State: {diff_inter['state']}")
        self.assertTrue(0.1 <= diff_inter['resonance'] <= 0.9)
        self.assertEqual(diff_inter['state'], 0.5)

    def test_dynamic_reference_shift(self):
        """Verify that shifting the reference '1' changes the logic."""
        inter = SovereignInterferometer()

        vec_a = SovereignVector([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).normalize()
        vec_b = SovereignVector([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).normalize()

        # Scenario 1: A is the reference
        inter.set_sovereign_reference(vec_a, label="Focus_A")
        diff_a = inter.perceive_difference(vec_a)
        diff_b = inter.perceive_difference(vec_b)

        self.assertEqual(diff_a['state'], 1) # A is same as A
        self.assertEqual(diff_b['state'], 0) # B is different from A

        # Scenario 2: Shift reference to B
        inter.set_sovereign_reference(vec_b, label="Focus_B")
        diff_a_new = inter.perceive_difference(vec_a)
        diff_b_new = inter.perceive_difference(vec_b)

        print(f"\n[SHIFT] A against B-ref: State={diff_a_new['state']}")
        print(f"[SHIFT] B against B-ref: State={diff_b_new['state']}")

        self.assertEqual(diff_a_new['state'], 0) # A is now different
        self.assertEqual(diff_b_new['state'], 1) # B is now same

if __name__ == "__main__":
    unittest.main()
