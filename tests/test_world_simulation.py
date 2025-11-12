
import unittest
from unittest.mock import MagicMock
import numpy as np

from Project_Sophia.core.world import World
from Project_Sophia.core.game_objects import Item
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

class TestWorldSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and WaveMechanics for testing."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        # Default mock for get_node to prevent ValueErrors on cell creation
        self.mock_kg_manager.get_node.return_value = {}
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_wave_mechanics.kg_manager = self.mock_kg_manager

    @unittest.skip("Skipping test related to old 'Law of Love' energy system.")
    def test_arc_reactor_energy_boost(self):
        """Test that the energy boost from the 'Law of Love' is affected by the love node's energy."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        self.mock_kg_manager.get_node.return_value = {'id': 'love', 'activation_energy': 2.0}
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.5
        initial_energy = 10.0
        world.add_cell('cell_A', initial_energy=initial_energy)
        world.run_simulation_step()
        cell_A_after_step1 = world.materialize_cell('cell_A')
        # Test that the energy has changed, reflecting the boost.
        self.assertNotAlmostEqual(cell_A_after_step1.energy, initial_energy, delta=0.01)

    def test_full_ecosystem_cycle(self):
        """Tests predation and celestial cycles working together with validated values."""
        def get_node_mock(node_id):
            if node_id == 'sun': return {'id': 'sun', 'activation_energy': 2.0}
            if node_id == 'love': return {'id': 'love', 'activation_energy': 0.0}
            return None
        self.mock_kg_manager.get_node.side_effect = get_node_mock
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.0

        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('plant_A', properties={'element_type': 'life'})
        world.add_cell('wolf_A', properties={'element_type': 'animal'})

        plant_idx = world.id_to_idx['plant_A']
        wolf_idx = world.id_to_idx['wolf_A']
        world.hp[plant_idx] = 10.0
        world.hp[wolf_idx] = 20.0
        world.day_length = 20 # for predictable cycle

        # --- Step 1: Day Time ---
        # time_step starts at 0. The first step will be time_step 1, which is 'day'.
        initial_plant_hp = world.hp[plant_idx]
        world.run_simulation_step()

        # Test the principle: Plant's HP should increase from sunlight
        self.assertGreater(world.hp[plant_idx], initial_plant_hp)

        # --- Step 2: Night Time ---
        # Manually set time_step so the next step is guaranteed to be night
        world.time_step = (world.day_length / 2) - 1 # Next step will be 10, which is the start of the night cycle
        hp_before_night_wolf = world.hp[wolf_idx]
        world.run_simulation_step()

        # Test the principle: Animal should lose some HP at night
        self.assertLess(world.hp[wolf_idx], hp_before_night_wolf)

    def test_starvation_system(self):
        """Tests that a cell starts losing HP when its hunger reaches zero."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('human_A')
        human_idx = world.id_to_idx['human_A']
        world.max_hunger[human_idx] = 100
        world.hunger[human_idx] = 10
        world.day_length = 20 # To make calculations predictable

        # Run simulation until hunger should be depleted
        # Hunger decay rate is max_hunger / (day_length * 3) = 100 / 60 = 1.666...
        # So, after 6 steps, hunger should be 10 - (6 * 1.666) approx 0.
        for _ in range(7):
            world.run_simulation_step()

        self.assertEqual(world.hunger[human_idx], 0)

        hp_after_hunger_depleted = world.hp[human_idx]
        world.run_simulation_step() # Run one more step to trigger starvation damage

        self.assertLess(world.hp[human_idx], hp_after_hunger_depleted)

    def test_stat_growth_from_action(self):
        """Tests that a cell's stat increases after performing a relevant action."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('wolf_A', properties={'element_type': 'animal', 'diet': 'carnivore'})
        world.add_cell('deer_A', properties={'element_type': 'animal', 'diet': 'herbivore'})
        world.add_connection('wolf_A', 'deer_A', 0.1) # Set a non-kin relationship strength

        wolf_idx = world.id_to_idx['wolf_A']
        world.diets[wolf_idx] = 'carnivore' # Explicitly set diet for test clarity
        initial_strength = world.strength[wolf_idx]

        # Position them close to ensure combat
        world.positions[wolf_idx] = np.array([0, 0, 0])
        world.positions[world.id_to_idx['deer_A']] = np.array([0.5, 0, 0])

        world.run_simulation_step()

        self.assertGreater(world.strength[wolf_idx], initial_strength)

    def test_power_system_awakening(self):
        """Tests that a cell awakens a power system when a stat reaches the threshold."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('warrior_A', properties={'element_type': 'animal', 'diet': 'carnivore'})
        warrior_idx = world.id_to_idx['warrior_A']

        # Set strength just below the awakening threshold
        world.strength[warrior_idx] = 49.9
        world.talent_strength[warrior_idx] = 0.5 # High talent for faster growth
        world.diets[warrior_idx] = 'carnivore'
        self.assertFalse(world.power_system_awakened[warrior_idx])

        # Create a target to trigger combat and stat growth
        world.add_cell('dummy_A', properties={'element_type': 'animal', 'diet': 'herbivore'})
        dummy_idx = world.id_to_idx['dummy_A']
        world.add_connection('warrior_A', 'dummy_A', 0.1) # Set a non-kin relationship strength
        world.positions[warrior_idx] = np.array([0, 0, 0])
        world.positions[dummy_idx] = np.array([0.1, 0, 0])

        # This step should cause combat, which increases strength beyond 50
        world.run_simulation_step()

        self.assertGreater(world.strength[warrior_idx], 50.0)
        self.assertEqual(world.power_system[warrior_idx], 'aura')
        self.assertTrue(world.power_system_awakened[warrior_idx])

    def test_civilization_lifecycle(self):
        """Tests the full cycle: farming, gathering, crafting, and harvesting."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())

        # 1. Setup
        world.add_cell('human_A', properties={'label': 'human', 'element_type': 'animal'})
        world.add_cell('forest_A', properties={'label': 'forest'})
        world.add_cell('earth_A', properties={'element_type': 'earth'})
        world.add_connection('human_A', 'forest_A', 0.1)
        world.add_connection('human_A', 'earth_A', 0.1)

        human_idx = world.id_to_idx['human_A']
        earth_idx = world.id_to_idx['earth_A']
        forest_idx = world.id_to_idx['forest_A']

        # Position them close, with earth being the closest
        world.positions[human_idx] = np.array([0, 0, 0])
        world.positions[earth_idx] = np.array([0.1, 0, 0])
        world.positions[forest_idx] = np.array([0.2, 0, 0])

        # Give stone for crafting
        world.inventories[human_idx].add_item(Item(name='stone', quantity=1))

        # --- AI should prioritize farming first because it's closer ---
        # 2. Till Land & Plant Seed
        # Run a few steps. Human should move to earth_A, till it, and plant.
        # This might happen very quickly.
        for _ in range(5):
            world.run_simulation_step()
        self.assertIn(world.farmland_state[earth_idx], ['tilled', 'planted'])

        # Keep running until the seed is planted
        for _ in range(5):
            if world.farmland_state[earth_idx] == 'planted': break
            world.run_simulation_step()
        self.assertEqual(world.farmland_state[earth_idx], 'planted')

        # 3. Gather Wood (while crop grows)
        # Now that farming is started, AI should look for other tasks.
        for _ in range(10):
            if 'wood' in world.inventories[human_idx].items: break
            world.run_simulation_step()
        self.assertIn('wood', world.inventories[human_idx].items)

        # 4. Craft stone axe
        # AI should now prioritize crafting as it has materials
        world.run_simulation_step()
        self.assertIn('stone_axe', world.inventories[human_idx].items)

        # 5. Grow and Harvest
        # Manually water and run simulation to advance crop growth
        for _ in range(15):
            world.water_level[earth_idx] = 10.0
            world.run_simulation_step()

        # Final step to trigger harvest AI
        world.run_simulation_step()
        self.assertIn('wheat', world.inventories[human_idx].items)


if __name__ == '__main__':
    unittest.main()
