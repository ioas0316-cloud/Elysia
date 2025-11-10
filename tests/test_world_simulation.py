import unittest
from unittest.mock import MagicMock

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestWorldSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and WaveMechanics for testing."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_wave_mechanics.kg_manager = self.mock_kg_manager

        # World requires primordial_dna and a logger
        self.world = World(
            primordial_dna={'instinct': 'test'},
            wave_mechanics=self.mock_wave_mechanics,
            logger=MagicMock()
        )

    def test_arc_reactor_energy_boost(self):
        """Test that the energy boost from the 'Law of Love' is affected by the love node's energy."""
        # --- SCENARIO 1: Love node has low energy ---
        # Setup: KG with a 'love' node having low activation_energy
        self.mock_kg_manager.get_node.return_value = {'id': 'love', 'activation_energy': 2.0}

        # Setup: WaveMechanics will return a fixed resonance value for our test cell
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5

        # Add a test cell to the world
        self.world.add_cell('cell_A', initial_energy=10.0)

        # Run one simulation step
        self.world.run_simulation_step()

        # Verification for Scenario 1
        # Expected boost = (resonance * love_energy * 0.5) + 0.5 (nurturing for isolated cell)
        # = (0.5 * 2.0 * 0.5) + 0.5 = 0.5 + 0.5 = 1.0
        # The cell starts with 10, gets the boost. We check if the final energy is close to 11.0
        cell_A_after_step1 = self.world.get_cell('cell_A')
        self.assertAlmostEqual(cell_A_after_step1.energy, 11.0, delta=0.1)

        # --- SCENARIO 2: Love node has high energy ---
        # Reset the world and cell for a clean test
        self.world = World(
            primordial_dna={'instinct': 'test'},
            wave_mechanics=self.mock_wave_mechanics,
            logger=MagicMock()
        )

        # Setup: Now the 'love' node has high activation_energy
        self.mock_kg_manager.get_node.return_value = {'id': 'love', 'activation_energy': 10.0}

        # Resonance value remains the same
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5

        # Add the same test cell
        self.world.add_cell('cell_A', initial_energy=10.0)

        # Run one simulation step
        self.world.run_simulation_step()

        # Verification for Scenario 2
        # Expected boost = (resonance * love_energy * 0.5) + 0.5 (nurturing)
        # = (0.5 * 10.0 * 0.5) + 0.5 = 2.5 + 0.5 = 3.0
        cell_A_after_step2 = self.world.get_cell('cell_A')
        self.assertAlmostEqual(cell_A_after_step2.energy, 13.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
