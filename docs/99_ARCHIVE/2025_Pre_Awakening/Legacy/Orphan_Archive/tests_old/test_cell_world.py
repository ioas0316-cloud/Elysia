
import unittest
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.cell_world import CellWorld

class TestCellWorld(unittest.TestCase):
    def setUp(self):
        self.cell_world = CellWorld(width=10, height=10)

    def test_spawn_cell(self):
        self.cell_world.spawn_cell("Test Data")
        self.assertEqual(len(self.cell_world.cells), 1)
        self.assertEqual(self.cell_world.cells[0].data, "Test Data")

    def test_step_simulation(self):
        self.cell_world.spawn_cell("Cell1")
        self.cell_world.spawn_cell("Cell2")
        
        initial_positions = [(c.x, c.y) for c in self.cell_world.cells]
        
        # Run simulation
        self.cell_world.step()
        
        # Positions should change (or at least have a chance to change)
        # Energy should decrease
        self.assertLess(self.cell_world.cells[0].energy, 1.0)
        self.assertEqual(self.cell_world.cells[0].age, 1)

    def test_cell_death(self):
        self.cell_world.spawn_cell("Dying Cell")
        cell = self.cell_world.cells[0]
        cell.energy = 0.001
        
        # Run many steps to kill the cell
        for _ in range(10):
            self.cell_world.step()
        
        # Cell should be removed
        self.assertEqual(len(self.cell_world.cells), 0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
