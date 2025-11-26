import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.organic_concept import ConceptVector
from Core.Evolution.organic_alchemy import OrganicAlchemy

class TestOrganicAlchemy(unittest.TestCase):
    def setUp(self):
        self.alchemy = OrganicAlchemy(dimension=64)

    def test_vector_creation(self):
        c = ConceptVector("Fire")
        self.assertEqual(len(c.vector), 64)
        self.assertAlmostEqual(np.linalg.norm(c.vector), 1.0, places=5)

    def test_combination(self):
        a = ConceptVector("Fire")
        b = ConceptVector("Water")
        
        c = self.alchemy.combine(a, b)
        self.assertIsNotNone(c)
        self.assertEqual(c.name, "Fire-Water")
        self.assertAlmostEqual(np.linalg.norm(c.vector), 1.0, places=5)
        
        # Check that result is somewhat related to parents (cosine distance < 1.0)
        dist_a = c.distance_to(a)
        dist_b = c.distance_to(b)
        print(f"Distance to A: {dist_a}, Distance to B: {dist_b}")
        
        # Since c is roughly (a+b)/2, it should be closer than orthogonal (dist=1.0)
        self.assertLess(dist_a, 1.0)
        self.assertLess(dist_b, 1.0)

    def test_annihilation(self):
        # Create opposite vectors
        a = ConceptVector("Matter")
        b = ConceptVector("AntiMatter")
        b.vector = -a.vector # Exact opposite
        
        c = self.alchemy.combine(a, b)
        self.assertIsNone(c, "Opposite vectors should annihilate")

if __name__ == '__main__':
    unittest.main()
