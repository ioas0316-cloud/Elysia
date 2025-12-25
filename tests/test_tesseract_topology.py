import unittest
import numpy as np
from Core.Cognition.Topology.tesseract_geometry import TesseractVector, TesseractGeometry
from Core.Cognition.Topology.fluid_intention import FluidIntention

class TestTesseractTopology(unittest.TestCase):
    def test_vector_normalization(self):
        v = TesseractVector(1, 0, 0, 0)
        norm = v.normalize()
        self.assertEqual(norm.magnitude(), 1.0)

        v2 = TesseractVector(2, 0, 0, 0)
        norm2 = v2.normalize()
        self.assertAlmostEqual(norm2.x, 1.0)

    def test_rotation_consistency(self):
        geo = TesseractGeometry()
        v = TesseractVector(1, 0, 0, 0)
        # Rotate 90 degrees in XW plane
        rotated = geo.rotate_xw(v, np.pi/2)
        # Should now be (0, 0, 0, 1) approx
        self.assertAlmostEqual(rotated.x, 0.0)
        self.assertAlmostEqual(rotated.w, 1.0)

    def test_fluid_resonance_point_focus(self):
        # Scale -> 0 (Sharp Focus)
        intention = FluidIntention(focus_w=0.5, scale=0.01)

        # Exact match
        self.assertAlmostEqual(intention.get_resonance_strength(0.5), 1.0)
        # Far away
        self.assertAlmostEqual(intention.get_resonance_strength(0.8), 0.0)

    def test_fluid_resonance_broad_focus(self):
        # Scale -> Large (Broad Focus)
        intention = FluidIntention(focus_w=0.5, scale=10.0)

        # Center is 1.0
        self.assertAlmostEqual(intention.get_resonance_strength(0.5), 1.0)
        # Far away is still high
        res_far = intention.get_resonance_strength(2.5)
        self.assertTrue(res_far > 0.9) # Should decay very slowly

if __name__ == '__main__':
    unittest.main()
