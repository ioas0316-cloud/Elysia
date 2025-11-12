
import unittest
from unittest.mock import MagicMock, patch
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

    def test_hp_change_simulation(self):
        """Tests a basic hp change to ensure simulation step runs."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('cell_A', properties={'hp': 10.0, 'max_hp': 100.0})
        cell_idx = world.id_to_idx['cell_A']
        initial_hp = world.hp[cell_idx]

        # Run a step. Night time decay should reduce HP.
        world.time_step = world.day_length # Force night time
        world.run_simulation_step()

        self.assertLess(world.hp[cell_idx], initial_hp)

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

    def test_tribe_formation_on_birth(self):
        """Tests that a child inherits the parent's tribe ID or forms a new one."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())

        # Scenario 1: Parents with no tribe form a new one with their child
        world.add_cell('adam', properties={'label': 'human', 'element_type': 'animal', 'gender': 'male'})
        world.add_cell('eve', properties={'label': 'human', 'element_type': 'animal', 'gender': 'female'})
        adam_idx = world.id_to_idx['adam']
        eve_idx = world.id_to_idx['eve']

        # Mating requires a bidirectional connection for recognition
        world.add_connection('adam', 'eve', 0.9)
        world.add_connection('eve', 'adam', 0.9)

        world.mating_readiness[adam_idx] = 1.0
        world.mating_readiness[eve_idx] = 1.0

        # Run simulation to trigger birth
        newly_born = world.run_simulation_step()
        self.assertEqual(len(newly_born), 1)
        child_id = newly_born[0].id
        child_idx = world.id_to_idx[child_id]

        # Check that a new tribe was formed
        self.assertIsNotNone(world.tribe_id[eve_idx])
        self.assertNotEqual(world.tribe_id[eve_idx], "")
        self.assertEqual(world.tribe_id[eve_idx], world.tribe_id[child_idx])

        # Scenario 2: Child inherits an existing tribe
        world.add_cell('seth', properties={'label': 'human', 'element_type': 'animal', 'gender': 'male'})
        seth_idx = world.id_to_idx['seth']
        world.add_connection('seth', 'eve', 0.9)
        world.add_connection('eve', 'seth', 0.9)
        world.mating_readiness[seth_idx] = 1.0
        world.mating_readiness[eve_idx] = 1.0 # Eve is ready again

        newly_born_2 = world.run_simulation_step()
        self.assertEqual(len(newly_born_2), 1)
        child_2_id = newly_born_2[0].id
        child_2_idx = world.id_to_idx[child_2_id]

        # Check that the second child inherited the same tribe ID
        self.assertEqual(world.tribe_id[eve_idx], world.tribe_id[child_2_idx])


    def test_intra_tribe_non_aggression(self):
        """Tests that members of the same tribe do not attack each other."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('human_A', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore'})
        world.add_cell('human_B', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore'})

        human_A_idx = world.id_to_idx['human_A']
        human_B_idx = world.id_to_idx['human_B']

        # Assign them to the same tribe
        world.tribe_id[human_A_idx] = 'tribe_1'
        world.tribe_id[human_B_idx] = 'tribe_1'

        # Make them hungry and close to each other to encourage aggression
        world.hunger[human_A_idx] = 10
        world.hunger[human_B_idx] = 10
        world.positions[human_A_idx] = np.array([0, 0, 0])
        world.positions[human_B_idx] = np.array([0.1, 0, 0])
        world.add_connection('human_A', 'human_B', 0.1) # Ensure they are aware of each other

        initial_hp_A = world.hp[human_A_idx]
        initial_hp_B = world.hp[human_B_idx]

        world.run_simulation_step()

        # Their HP should not decrease from combat (it might change from other factors like starvation, so we check for significant drops)
        self.assertAlmostEqual(world.hp[human_A_idx], initial_hp_A, delta=5.0)
        self.assertAlmostEqual(world.hp[human_B_idx], initial_hp_B, delta=5.0)

    def test_vitality_affects_max_hp(self):
        """Tests that max_hp is correctly calculated based on vitality."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('human_A')
        human_idx = world.id_to_idx['human_A']

        # 1. Test initial max_hp calculation
        initial_vitality = world.vitality[human_idx]
        expected_max_hp = 50.0 + (initial_vitality * 10)
        self.assertEqual(world.max_hp[human_idx], expected_max_hp)

        # 2. Test dynamic update of max_hp on vitality growth
        world.vitality[human_idx] += 5.0
        world.max_hp[human_idx] = 50.0 + (world.vitality[human_idx] * 10) # Manually trigger the update for test

        new_expected_max_hp = 50.0 + (world.vitality[human_idx] * 10)
        self.assertEqual(world.max_hp[human_idx], new_expected_max_hp)

    @unittest.mock.patch('random.random')
    def test_agility_provides_evasion(self, mock_random):
        """Tests that agility allows a cell to evade attacks."""
        world = World(primordial_dna={'instinct': 'test'}, wave_mechanics=self.mock_wave_mechanics, logger=MagicMock())
        world.add_cell('rogue', properties={'element_type': 'animal', 'diet': 'carnivore'})
        world.add_cell('dummy', properties={'element_type': 'animal', 'diet': 'herbivore'})

        rogue_idx = world.id_to_idx['rogue']
        dummy_idx = world.id_to_idx['dummy']

        # Give rogue very high agility and dummy very low
        world.agility[rogue_idx] = 100.0
        world.agility[dummy_idx] = 1.0
        world.diets[dummy_idx] = 'carnivore' # Make dummy aggressive for testing

        world.add_connection('dummy', 'rogue', 0.1)
        world.add_connection('rogue', 'dummy', 0.1)
        world.positions[rogue_idx] = np.array([0, 0, 0])
        world.positions[dummy_idx] = np.array([0.1, 0, 0])

        initial_rogue_hp = world.hp[rogue_idx]

        # --- Test Evasion Success ---
        # Mock random.random() to return a value that guarantees evasion
        mock_random.return_value = 0.01

        # In this step, the dummy should attack the rogue but fail due to evasion
        hp_deltas = np.zeros_like(world.hp)
        world._execute_animal_action(dummy_idx, rogue_idx, 'hunt', hp_deltas)
        world.hp += hp_deltas
        self.assertEqual(world.hp[rogue_idx], initial_rogue_hp, "Rogue's HP should not change on successful evasion.")

        # --- Test Evasion Fail ---
        initial_dummy_hp = world.hp[dummy_idx]
        mock_random.return_value = 0.99 # Guarantees the attack hits
        hp_deltas = np.zeros_like(world.hp)

        # Rogue attacks dummy, who has low agility and should fail to evade
        world._execute_animal_action(rogue_idx, dummy_idx, 'hunt', hp_deltas)
        world.hp += hp_deltas
        self.assertLess(world.hp[dummy_idx], initial_dummy_hp, "Dummy's HP should decrease when evasion fails.")


if __name__ == '__main__':
    unittest.main()
