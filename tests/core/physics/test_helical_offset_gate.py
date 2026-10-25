"""
Unit tests for HelicalOffsetTorusGate and CryptographicPhaseMatcher
(tests/core/physics/test_helical_offset_gate.py)
"""

import unittest
import numpy as np
from L1_causal_field.core.physics.helical_offset_gate import (
    HelicalOffsetTorusGate,
    CryptographicPhaseMatcher
)

class TestHelicalOffsetTorusGate(unittest.TestCase):
    def setUp(self):
        self.gate = HelicalOffsetTorusGate(
            major_radius=3.0,
            minor_radius=1.0,
            helix_pitch=5.0,
            max_energy=100.0,
            scale_sigma=0.2
        )
        self.matcher = CryptographicPhaseMatcher(gate=self.gate)

    def test_toroidal_helix_coordinates(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        X, Y, Z = self.gate.compute_toroidal_helix(theta, phase_offset=0.0)

        self.assertEqual(X.shape, (100,))
        self.assertEqual(Y.shape, (100,))
        self.assertEqual(Z.shape, (100,))

        # Test 3D bounds
        max_r = self.gate.R + self.gate.r
        self.assertTrue(np.all(np.abs(X) <= max_r + 1e-5))
        self.assertTrue(np.all(np.abs(Y) <= max_r + 1e-5))
        self.assertTrue(np.all(np.abs(Z) <= self.gate.r + 1e-5))

    def test_phase_difference_normalization(self):
        # Difference between 0.1 and 0.1 is 0.0
        diff1 = self.gate.compute_phase_diff(0.1, 0.1)
        self.assertAlmostEqual(diff1, 0.0, places=5)

        # Cyclic wrap-around check: pi vs -pi is 0 or 2*pi equivalent (minimal angle = 0)
        diff2 = self.gate.compute_phase_diff(np.pi, -np.pi)
        self.assertAlmostEqual(abs(diff2), 0.0, places=5)

    def test_gate_barrier_energy_and_permeability(self):
        # Perfect phase alignment (Delta phi = 0) -> E_gate = 0, Permeability = 1.0
        e_zero = self.gate.compute_gate_barrier_energy(0.0)
        p_one = self.gate.compute_permeability(0.0)
        self.assertAlmostEqual(e_zero, 0.0, places=5)
        self.assertAlmostEqual(p_one, 1.0, places=5)

        # Mismatched phase (Delta phi = 1.0 rad) -> E_gate > 0, Permeability < 0.1
        e_mismatch = self.gate.compute_gate_barrier_energy(1.0)
        p_mismatch = self.gate.compute_permeability(1.0)
        self.assertGreater(e_mismatch, 90.0)
        self.assertLess(p_mismatch, 0.1)

    def test_linking_gate_evaluation(self):
        # Matching phases link
        res_link = self.gate.evaluate_linking_gate(0.5, 0.5)
        self.assertTrue(res_link["is_linked"])
        self.assertAlmostEqual(res_link["gate_energy"], 0.0, places=5)
        self.assertAlmostEqual(res_link["permeability"], 1.0, places=5)

        # Mismatched phases fail to link
        res_fail = self.gate.evaluate_linking_gate(0.0, 1.5)
        self.assertFalse(res_fail["is_linked"])

    def test_cryptographic_phase_decryption(self):
        payload = {"secret": "Elysia_Topological_Truth"}
        payload_hash = 123456
        cipher_phase = self.matcher.encode_payload_to_phase(payload_hash)

        # Correct key alignment
        dec_success = self.matcher.decrypt_topological_gate(
            cipher_phase=cipher_phase,
            key_phase=cipher_phase,
            payload_data=payload
        )
        self.assertTrue(dec_success["success"])
        self.assertEqual(dec_success["decrypted_payload"], payload)
        self.assertAlmostEqual(dec_success["barrier_energy"], 0.0, places=5)

        # Incorrect key phase
        wrong_key_phase = cipher_phase + 0.8
        dec_fail = self.matcher.decrypt_topological_gate(
            cipher_phase=cipher_phase,
            key_phase=wrong_key_phase,
            payload_data=payload
        )
        self.assertFalse(dec_fail["success"])
        self.assertIsNone(dec_fail["decrypted_payload"])
        self.assertGreater(dec_fail["barrier_energy"], 50.0)

if __name__ == "__main__":
    unittest.main()
