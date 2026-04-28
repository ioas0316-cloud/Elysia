import sys
import os
import unittest

# Add root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.sovereign_math import SovereignVector, FogField, PrismaticRefractor

class TestPrismaticCognition(unittest.TestCase):
    def test_fog_energy_accumulation(self):
        fog = FogField(capacity=100.0)

        # High resonance, high complexity -> Low accumulation
        fog.accumulate_mist(resonance=0.9, complexity=1.0)
        self.assertLess(fog.fog_energy, 1.0)

        # Low resonance (Unknown), high complexity -> High accumulation
        for _ in range(10):
            fog.accumulate_mist(resonance=0.1, complexity=10.0)
        print(f"\n[FOG] Energy after multiple unknown encounters: {fog.fog_energy:.2f}")
        self.assertGreater(fog.fog_energy, 5.0)

        # Test Leap readiness
        fog.fog_energy = 90.0
        self.assertTrue(fog.can_leap())

        intensity = fog.discharge_leap()
        self.assertEqual(intensity, 0.9)
        self.assertEqual(fog.fog_energy, 9.0) # Residue

    def test_prismatic_refraction(self):
        refractor = PrismaticRefractor()

        # Create a non-uniform vector
        data = [0.1] * 21
        data[0:3] = [1.0, 1.0, 1.0] # 'Red' band
        data[18:21] = [0.8, 0.8, 0.8] # 'Violet' band

        vec = SovereignVector(data)
        spectrum = refractor.refract(vec)

        print(f"[SPECTRUM] Refracted spectrum: {spectrum}")
        self.assertIn("RED", spectrum)
        self.assertIn("VIOLET", spectrum)
        self.assertGreater(spectrum["RED"], spectrum["YELLOW"])
        self.assertGreater(spectrum["VIOLET"], spectrum["YELLOW"])

if __name__ == "__main__":
    unittest.main()
