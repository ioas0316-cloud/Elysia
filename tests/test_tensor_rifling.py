
import unittest
from Project_Sophia.core.tensor_wave import Tensor3D, FrequencyWave, propagate_wave
import numpy as np

class TestTensorRifling(unittest.TestCase):

    def test_tensor_initialization(self):
        t = Tensor3D(1.0, 0.5, 0.2)
        self.assertAlmostEqual(t.structure, 1.0)
        self.assertAlmostEqual(t.emotion, 0.5)
        self.assertAlmostEqual(t.identity, 0.2)
        self.assertAlmostEqual(t.spin_magnitude(), 0.0)

    def test_rifling_calculation(self):
        # Intent Vector along Z (Identity)
        t = Tensor3D(0.0, 0.0, 1.0, spin=np.array([0.0, 0.0, 1.0]))

        # Spin is aligned with Intent -> High Rifling
        rifling = t.calculate_rifling()
        self.assertAlmostEqual(rifling, 1.0)

        # Spin is orthogonal to Intent -> Zero Rifling (Tumbling)
        t_tumble = Tensor3D(0.0, 0.0, 1.0, spin=np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(t_tumble.calculate_rifling(), 0.0)

        # Spin is opposed -> Negative Rifling (Braking)
        t_brake = Tensor3D(0.0, 0.0, 1.0, spin=np.array([0.0, 0.0, -1.0]))
        self.assertAlmostEqual(t_brake.calculate_rifling(), -1.0)

    def test_frequency_generates_spin(self):
        # Low frequency -> No spin
        low_t = Tensor3D.distribute_frequency(50.0)
        self.assertAlmostEqual(low_t.spin_magnitude(), 0.0)

        # High frequency -> Auto spin
        high_t = Tensor3D.distribute_frequency(600.0)
        self.assertGreater(high_t.spin_magnitude(), 0.0)
        # Spin should be along Z for high freq
        self.assertGreater(high_t.spin[2], 0.0)

    def test_spin_induction(self):
        source = Tensor3D(0, 0, 1, spin=np.array([0, 0, 10])) # High spin
        target = Tensor3D(0, 0, 1) # No spin

        result = propagate_wave(source, target)

        # Target should have acquired some spin
        self.assertGreater(result.spin_magnitude(), 0.0)
        # Check direction match roughly
        spin_dir = result.spin / result.spin_magnitude()
        expected_dir = np.array([0, 0, 1])
        self.assertTrue(np.allclose(spin_dir, expected_dir, atol=0.1))

if __name__ == '__main__':
    unittest.main()
