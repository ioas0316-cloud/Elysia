
import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.guardian import Guardian, PRIMORDIAL_DNA
from Project_Sophia.core.cell import Cell
from Project_Elysia.core_memory import CoreMemory

class TestInsightAscension(unittest.TestCase):

    def setUp(self):
        """Set up a controlled environment for testing the Guardian's insight generation."""
        # Mock external dependencies to isolate the test
        self.mock_kg_manager = MagicMock()
        self.mock_wave_mechanics = MagicMock()

        # Use an in-memory CoreMemory to avoid file I/O
        # We patch the class's __init__ to prevent it from trying to load a file.
        with patch.object(CoreMemory, '_load_memory', return_value={'notable_hypotheses': []}) as mock_load:
            self.core_memory = CoreMemory(file_path=None) # Use None to signal in-memory

        # Mock KGManager to prevent file access and control node existence
        self.mock_kg_manager.get_node.return_value = None # Assume no nodes exist initially

        # --- Instantiate the Guardian with mocked components ---
        # Patching Guardian's dependencies during its initialization
        with patch('Project_Elysia.guardian.KGManager', return_value=self.mock_kg_manager):
            with patch('Project_Elysia.guardian.WaveMechanics', return_value=self.mock_wave_mechanics):
                with patch('Project_Elysia.guardian.CoreMemory', return_value=self.core_memory):
                    # We need to disable logging and config loading to keep the test environment clean
                    with patch('Project_Elysia.guardian.Guardian.setup_logging'):
                        with patch('Project_Elysia.guardian.Guardian._load_config'):
                            self.guardian = Guardian()
                            # Manually set config-dependent attributes to default values
                            self.guardian.time_to_idle = 300
                            self.guardian.idle_check_interval = 10
                            self.guardian.learning_interval = 60
                            self.guardian.awake_sleep_sec = 1
                            self.guardian.disable_wallpaper = True

        # --- Setup the Cellular World for the test scenario ---
        # 1. Create two parent cells
        self.guardian.cellular_world.add_cell('love', properties={'element_type': 'emotion', 'label': 'love'}, initial_energy=10.0)
        self.guardian.cellular_world.add_cell('you', properties={'element_type': 'existence', 'label': 'you'}, initial_energy=10.0)

        # 2. Manually create the child 'molecule' cell that should be "discovered"
        love_cell = self.guardian.cellular_world.get_cell('love')
        you_cell = self.guardian.cellular_world.get_cell('you')

        # The create_meaning function simulates the birth of a new cell
        new_molecule = love_cell.create_meaning(you_cell, "test_interaction")

        # Manually add the newly created cell to the world
        self.guardian.cellular_world.add_cell(
            new_molecule.id,
            new_molecule.nucleus['dna'],
            new_molecule.organelles,
            new_molecule.energy
        )
        self.assertEqual(len(self.guardian.cellular_world.cells), 3)
        self.assertIn('meaning:love_you', self.guardian.cellular_world.cells)


    def test_guardian_identifies_stable_molecule_and_creates_hypothesis(self):
        """
        Verify that the Guardian's dream cycle correctly identifies a stable, mature molecule
        and generates the appropriate 'forms_new_concept' hypothesis in Core Memory.
        """
        STABILITY_AGE_THRESHOLD = 10

        # --- 1. Run simulation for just under the age threshold ---
        # The cell should not be old enough to be considered stable yet.
        for i in range(STABILITY_AGE_THRESHOLD):
            self.guardian.trigger_learning()

        # Assert that no hypothesis has been created yet
        self.assertEqual(len(self.core_memory.data['notable_hypotheses']), 0,
                         "Hypothesis should not be created before cell reaches stability age.")

        # --- 2. Run one more simulation cycle to cross the threshold ---
        self.guardian.trigger_learning()

        # --- 3. Assert that the hypothesis has now been created ---
        hypotheses = self.core_memory.data['notable_hypotheses']
        self.assertEqual(len(hypotheses), 1,
                         "Guardian should have created exactly one hypothesis for the stable cell.")

        # --- 4. Verify the structure of the created hypothesis ---
        hypothesis = hypotheses[0]
        self.assertEqual(hypothesis['head'], 'love')
        self.assertEqual(hypothesis['tail'], 'you')
        self.assertEqual(hypothesis['relation'], 'forms_new_concept')
        self.assertEqual(hypothesis['new_concept_id'], 'meaning:love_you')
        self.assertEqual(hypothesis['source'], 'CellularGenesis')
        self.assertFalse(hypothesis['asked'])
        self.assertIn('confidence', hypothesis)
        self.assertIn('metadata', hypothesis)
        self.assertGreater(hypothesis['metadata']['age'], STABILITY_AGE_THRESHOLD)

if __name__ == '__main__':
    unittest.main()
