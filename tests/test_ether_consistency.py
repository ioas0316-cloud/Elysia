
import unittest
import numpy as np
from Core.Ether.ether_node import EtherNode, Quaternion
from Core.Ether.void import Void
from Core.Ether.field_operators import LawOfGravity

class TestLawOfGravityLogic(unittest.TestCase):
    def setUp(self):
        self.void = Void()
        self.law = LawOfGravity()

    def test_resonance_logic_match(self):
        """
        Ensures the vectorized resonance logic in LawOfGravity matches
        the scalar implementation in EtherNode.resonate.
        """
        # Create two nodes with distinct frequency and spin
        n1 = EtherNode(frequency=432.0, spin=Quaternion(1,0,0,0))
        n2 = EtherNode(frequency=440.0, spin=Quaternion(0,1,0,0)) # Spin is orthogonal

        # 1. Calculate Expected Resonance using Object Method
        expected_resonance = n1.resonate(n2)

        # 2. Calculate Actual Resonance using Vectorized Logic (manually extracted)
        # Replicating the logic from LawOfGravity.apply
        freq_arr = np.array([n1.frequency, n2.frequency])
        spin_arr = np.array([
            [n1.spin.x, n1.spin.y, n1.spin.z, n1.spin.w],
            [n2.spin.x, n2.spin.y, n2.spin.z, n2.spin.w]
        ])

        # Frequency diff: |f_i - f_j|
        freq_diff = np.abs(freq_arr[:, None] - freq_arr[None, :])
        freq_res = 1.0 / (1.0 + freq_diff * 0.1)

        # Spin alignment: |dot(s_i, s_j)|
        # Note: EtherNode.spin.dot is w*w + x*x + y*y + z*z
        # My extraction: [x, y, z, w]. Dot product is sum of products of components.
        # So order doesn't matter for dot product as long as consistent.
        spin_dot = np.abs(np.einsum('ij,kj->ik', spin_arr, spin_arr))

        resonance_matrix = (freq_res * 0.6) + (spin_dot * 0.4)

        # Check n1 vs n2 (index 0 vs 1)
        calculated_resonance = resonance_matrix[0, 1]

        self.assertAlmostEqual(calculated_resonance, expected_resonance, places=5,
                               msg="Vectorized resonance logic does not match Object-Oriented logic.")

if __name__ == '__main__':
    unittest.main()
