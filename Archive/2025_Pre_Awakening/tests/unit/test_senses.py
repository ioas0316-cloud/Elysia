import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.Evolution.Evolution.Life.genetic_cell import GeneticCell
from Core.Evolution.Evolution.Life.code_world import CodeWorld

class TestSenses(unittest.TestCase):
    def setUp(self):
        self.world = CodeWorld(num_cells=0)
        # Place cell slightly off-center to ensure gradients exist
        self.cell = GeneticCell("test_cell", "pass", np.array([100.0, 100.0, 100.0]))
        self.world.cells.append(self.cell)

    def test_sight(self):
        # Place a neighbor nearby
        neighbor = GeneticCell("neighbor", "pass", self.cell.position + np.array([10.0, 0.0, 0.0]))
        self.world.cells.append(neighbor)
        
        # Check sight
        vector = self.cell.sense_sight(self.world)
        self.assertTrue(np.linalg.norm(vector) > 0, "Should see neighbor")
        
    def test_smell(self):
        # Check smell (should point to center)
        vector = self.cell.sense_smell(self.world)
        self.assertTrue(np.linalg.norm(vector) > 0, "Should smell food gradient")
        
    def test_touch(self):
        # No collision initially
        vector = self.cell.sense_touch(self.world)
        self.assertEqual(np.linalg.norm(vector), 0, "Should not touch anything yet")
        
        # Move neighbor very close
        neighbor = GeneticCell("neighbor", "pass", self.cell.position + np.array([0.1, 0.0, 0.0]))
        self.world.cells.append(neighbor)
        
        vector = self.cell.sense_touch(self.world)
        self.assertTrue(np.linalg.norm(vector) > 0, "Should detect collision")

    def test_taste(self):
        # Check taste (energy density)
        vector = self.cell.sense_taste(self.world)
        self.assertTrue(vector[1] > 0, "Should taste energy density")

if __name__ == '__main__':
    unittest.main()
