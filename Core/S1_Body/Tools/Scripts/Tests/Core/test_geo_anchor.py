import unittest
import math
from Core.Physiology.Sensory.Spatial.geo_anchor import GeoMagneticAnchor

class TestGeoMagneticAnchor(unittest.TestCase):
    def setUp(self):
        self.anchor = GeoMagneticAnchor()

    def test_calculate_magnetic_flux(self):
        # Test Equator (Lat 0, Lon 0)
        flux = self.anchor.calculate_magnetic_flux(0, 0)
        self.assertIn('x', flux)
        self.assertIn('y', flux)
        self.assertIn('z', flux)
        self.assertIn('intensity', flux)

        # At equator, intensity should be roughly B0 (31000 nT)
        self.assertTrue(30000 <= flux['intensity'] <= 33000)

    def test_phase_signature_consistency(self):
        # The same location should produce the same signature
        lat, lon = 37.5665, 126.9780
        sig1 = self.anchor.get_phase_signature(lat, lon)
        sig2 = self.anchor.get_phase_signature(lat, lon)

        self.assertEqual(sig1['frequency'], sig2['frequency'])
        self.assertEqual(sig1['phase'], sig2['phase'])
        self.assertEqual(sig1['vector'], sig2['vector'])

    def test_resonance_logic(self):
        # Two identical locations should have perfect resonance (1.0)
        sig = self.anchor.get_phase_signature(37.5, 127.0)
        resonance = self.anchor.calculate_resonance(sig, sig)
        self.assertAlmostEqual(resonance, 1.0, places=4)

        # Two very different locations should have lower resonance
        sig_a = self.anchor.get_phase_signature(37.5, 127.0) # Seoul ish
        sig_b = self.anchor.get_phase_signature(40.7, -74.0) # NYC ish

        resonance_diff = self.anchor.calculate_resonance(sig_a, sig_b)
        self.assertTrue(resonance_diff < 1.0)
        self.assertTrue(resonance_diff >= 0.0)

if __name__ == '__main__':
    unittest.main()
