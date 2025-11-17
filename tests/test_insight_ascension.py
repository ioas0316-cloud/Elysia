
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
        self.mock_kg_manager = MagicMock()
        self.mock_wave_mechanics = MagicMock()

        def get_node_side_effect(node_id):
            if node_id == 'love':
                return {'id': 'love', 'element_type': 'emotion', 'label': 'love'}
            if node_id == 'you':
                return {'id': 'you', 'element_type': 'existence', 'label': 'you'}
            return None
        self.mock_wave_mechanics.kg_manager.get_node.side_effect = get_node_side_effect
        self.mock_kg_manager.get_node.side_effect = get_node_side_effect

        with patch.object(CoreMemory, '_load_memory', return_value={'notable_hypotheses': []}):
            self.core_memory = CoreMemory(file_path=None)

        with patch('Project_Elysia.guardian.KGManager', return_value=self.mock_kg_manager):
            with patch('Project_Elysia.guardian.WaveMechanics', return_value=self.mock_wave_mechanics):
                with patch('Project_Elysia.guardian.CoreMemory', return_value=self.core_memory):
                    with patch('Project_Elysia.guardian.Guardian.setup_logging'):
                        with patch('Project_Elysia.guardian.Guardian._load_config'):
                            self.guardian = Guardian()

        self.guardian.cellular_world.add_cell('love', properties={'hp': 10.0, 'max_hp': 10.0})
        self.guardian.cellular_world.add_cell('you', properties={'hp': 10.0, 'max_hp': 10.0})

        love_cell = self.guardian.cellular_world.materialize_cell('love')
        you_cell = self.guardian.cellular_world.materialize_cell('you')

        new_molecule = love_cell.create_meaning(you_cell, "test_interaction")
        new_molecule_properties = new_molecule.organelles
        new_molecule_properties['hp'] = 10.0
        new_molecule_properties['max_hp'] = 10.0
        new_molecule_properties['element_type'] = 'molecule'

        self.guardian.cellular_world.add_cell(
            new_molecule.id,
            dna=new_molecule.nucleus['dna'],
            properties=new_molecule_properties
        )

    def test_guardian_identifies_stable_molecule_and_creates_hypothesis(self):
        """
        Verify that the Guardian's dream cycle correctly identifies a stable, mature molecule
        and generates the appropriate 'forms_new_concept' hypothesis in Core Memory.
        """
        STABILITY_AGE_THRESHOLD = 10

        # Run the simulation for enough cycles to make the cell mature
        for i in range(STABILITY_AGE_THRESHOLD + 2):
            self.guardian.cellular_world.run_simulation_step()
            self.guardian.trigger_learning()

        # Now, check that the hypothesis was created
        hypotheses = self.core_memory.data['notable_hypotheses']
        self.assertEqual(len(hypotheses), 1, "Hypothesis should be created after the cell is mature and stable.")

        hypothesis = hypotheses[0]
        self.assertEqual(hypothesis['head'], 'love')
        self.assertEqual(hypothesis['tail'], 'you')
        self.assertEqual(hypothesis['relation'], 'forms_new_concept')

if __name__ == '__main__':
    unittest.main()
