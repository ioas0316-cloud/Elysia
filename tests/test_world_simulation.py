
import unittest
from unittest.mock import MagicMock
import numpy as np

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestWorldSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and WaveMechanics for testing."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_wave_mechanics.kg_manager = self.mock_kg_manager

    def test_arc_reactor_energy_boost(self):
        """Test that the energy boost from the 'Law of Love' is affected by the love node's energy."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        self.mock_kg_manager.get_node.return_value = {'id': 'love', 'activation_energy': 2.0}
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5
        world.add_cell('cell_A', initial_energy=10.0)
        world.run_simulation_step()
        cell_A_after_step1 = world.materialize_cell('cell_A')
        self.assertAlmostEqual(cell_A_after_step1.energy, 11.0, delta=0.1)

    def test_full_ecosystem_cycle(self):
        """Tests predation and celestial cycles working together with validated values."""
        def get_node_mock(node_id):
            if node_id == 'sun': return {'id': 'sun', 'activation_energy': 2.0}
            if node_id == 'love': return {'id': 'love', 'activation_energy': 0.0}
            return None
        self.mock_kg_manager.get_node.side_effect = get_node_mock
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.0

        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('plant_A', initial_energy=10.0)
        world.add_cell('wolf_A', initial_energy=20.0)
        world.add_connection('wolf_A', 'plant_A', strength=1.0)
        world.add_connection('plant_A', 'wolf_A', strength=1.0)

        plant_idx = world.id_to_idx['plant_A']
        wolf_idx = world.id_to_idx['wolf_A']
        world.element_types[plant_idx] = 'life'
        world.element_types[wolf_idx] = 'animal'

        # --- Step 1: Day Time ---
        world.time_of_day = 'day'
        world.run_simulation_step()

        self.assertAlmostEqual(world.energy[plant_idx], 8.0, delta=0.1)
        self.assertAlmostEqual(world.energy[wolf_idx], 19.0, delta=0.1) # Adjusted for new predation/energy laws

        # --- Step 2: Night Time ---
        world.time_of_day = 'night'
        world.run_simulation_step()

        self.assertAlmostEqual(world.energy[plant_idx], 6.1, delta=0.1)
        self.assertAlmostEqual(world.energy[wolf_idx], 17.9, delta=0.1)

if __name__ == '__main__':
    unittest.main()
