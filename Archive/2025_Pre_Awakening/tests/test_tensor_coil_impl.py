
import unittest
import math
import sys
import os

# Add repo root to path to import Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Memory.tensor_field import TensorCoil, TensorWell

class TestTensorCoil(unittest.TestCase):
    def test_initialization(self):
        coil = TensorCoil()
        self.assertIsNotNone(coil)
        self.assertEqual(coil.wells, [])

    def test_add_well(self):
        coil = TensorCoil()
        tensor = TensorCoil.create_isotropic_tensor(-1.0)
        coil.add_well(0, 0, 0, 0, tensor, "center")
        self.assertEqual(len(coil.wells), 1)
        self.assertEqual(coil.wells[0].label, "center")

    def test_isotropic_field(self):
        # Test simple attraction
        coil = TensorCoil(smoothing_radius=0.0) # Disable smoothing for exact math check
        # Force = -1 * vec / r^3
        # At (1,0,0,0), r=1. vec=(1,0,0,0). Force should be (-1, 0, 0, 0)

        tensor = TensorCoil.create_isotropic_tensor(-1.0)
        coil.add_well(0, 0, 0, 0, tensor)

        fx, fy, fz, fw = coil.get_field_at(1, 0, 0, 0)
        self.assertAlmostEqual(fx, -1.0)
        self.assertAlmostEqual(fy, 0.0)

        # At (2,0,0,0), r=2. vec=(2,0,0,0). r^3=8. Force = -1 * (2,0,0,0) / 8 = (-0.25, 0, 0, 0)
        fx, fy, fz, fw = coil.get_field_at(2, 0, 0, 0)
        self.assertAlmostEqual(fx, -0.25)

    def test_rotational_field(self):
        # Test rotation in XY plane
        coil = TensorCoil(smoothing_radius=0.0)
        # Tensor:
        # [0 -1 0 0]
        # [1  0 0 0]
        # ...
        # At (1,0,0,0), d=(1,0,0,0). T*d = (0, 1, 0, 0).
        # Force = (0, 1, 0, 0) / 1^3 = (0, 1, 0, 0)

        tensor = TensorCoil.create_rotation_tensor((0, 1), strength=1.0)
        coil.add_well(0, 0, 0, 0, tensor)

        fx, fy, fz, fw = coil.get_field_at(1, 0, 0, 0)
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, 1.0)

        # At (0,1,0,0), d=(0,1,0,0). T*d = (-1, 0, 0, 0).
        fx, fy, fz, fw = coil.get_field_at(0, 1, 0, 0)
        self.assertAlmostEqual(fx, -1.0)
        self.assertAlmostEqual(fy, 0.0)

    def test_combined_field(self):
        # Spiral: Attraction + Rotation
        coil = TensorCoil(smoothing_radius=0.0)
        tensor = TensorCoil.create_combined_tensor(attraction=-1.0, rotation_planes=[((0,1), 1.0)])
        coil.add_well(0, 0, 0, 0, tensor)

        # At (1,0,0,0)
        # Attraction part: (-1, 0, 0, 0)
        # Rotation part: (0, 1, 0, 0)
        # Result: (-1, 1, 0, 0)

        fx, fy, fz, fw = coil.get_field_at(1, 0, 0, 0)
        self.assertAlmostEqual(fx, -1.0)
        self.assertAlmostEqual(fy, 1.0)

if __name__ == '__main__':
    unittest.main()
