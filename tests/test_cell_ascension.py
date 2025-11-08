import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.guardian import Guardian
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core.cell import Cell

class TestCellAscension(unittest.TestCase):

    def setUp(self):
        """Set up a clean, integrated environment for testing the full ascension loop."""
        # --- File Paths for Test Data ---
        self.test_kg_path = 'data/test_kg_ascension.json'
        self.test_memory_path = 'data/test_core_memory_ascension.json'

        # --- Clean up old test files ---
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

        # --- Instantiate Core Components ---
        # We use real components for this integration test
        from tools.kg_manager import KGManager
        from Project_Elysia.core_memory import CoreMemory
        from Project_Sophia.core.world import World

        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.core_memory = CoreMemory(file_path=self.test_memory_path)
        from Project_Elysia.guardian import PRIMORDIAL_DNA
        self.cellular_world = World(primordial_dna=PRIMORDIAL_DNA)

        # --- Mock External Dependencies and Instantiate Guardian & Pipeline ---
        with patch('Project_Elysia.guardian.WebSearchCortex'), \
             patch('Project_Elysia.guardian.KnowledgeDistiller'), \
             patch('Project_Elysia.guardian.ExplorationCortex'), \
             patch('Project_Sophia.inquisitive_mind.InquisitiveMind'):

            self.guardian = Guardian()
            # Inject our real components into the Guardian
            self.guardian.kg_manager = self.kg_manager
            self.guardian.core_memory = self.core_memory
            self.guardian.cellular_world = self.cellular_world

            self.pipeline = CognitionPipeline(cellular_world=self.cellular_world)
            # Inject the same components into the Pipeline to ensure shared state
            self.pipeline.kg_manager = self.kg_manager
            self.pipeline.core_memory = self.core_memory

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def test_full_ascension_process(self):
        """
        Verify the entire Cell Ascension process:
        1. A 'meaning' cell is identified as an ascension candidate.
        2. An ascension hypothesis is created by the Guardian.
        3. The pipeline asks the user for approval.
        4. The user approves the ascension.
        5. A new node is created in the Knowledge Graph.
        """
        # --- Step 1 & 2: A 'meaning' cell is born and identified ---
        # Manually create a "newly born" cell that is a candidate for ascension
        from Project_Elysia.guardian import PRIMORDIAL_DNA
        ascension_candidate_cell = Cell(
            concept_id="meaning:사랑_성장",
            dna=PRIMORDIAL_DNA,
            initial_properties={"parents": ["사랑", "성장"]}
        )

        # Directly call the handler to simulate the Guardian's discovery
        self.guardian._handle_ascension_candidates([ascension_candidate_cell])

        # Verify that the ascension hypothesis was created in Core Memory
        hypotheses = self.core_memory.get_unasked_hypotheses()
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0]['relation'], '승천')
        self.assertEqual(hypotheses[0]['head'], 'meaning:사랑_성장')

        # --- Step 3: The pipeline asks for approval ---
        # The pipeline's hypothesis check should now pick up the ascension hypothesis
        question = self.pipeline._check_and_verify_hypotheses("아무 말") # Any message triggers the check

        self.assertIsNotNone(question)
        self.assertIn("새로운 의미 'meaning:사랑_성장'가 탄생했습니다.", question)
        self.assertIn("영원한 '개념'으로 승천시켜 지식의 일부로 만들까요?", question)

        # Verify that a hypothesis is now pending
        self.assertIsNotNone(self.pipeline.pending_hypothesis)

        # --- Step 4 & 5: The user approves, and the node is created ---
        # Now, send an approval message
        approval_message = "응, 승인할게"
        response = self.pipeline._check_and_verify_hypotheses(approval_message)

        self.assertIn("새로운 개념 'meaning:사랑_성장'이(가) 지식의 일부로 승천했습니다.", response)

        # --- Step 6: Final Verification ---
        # Check that the new node exists in the Knowledge Graph
        new_node = self.kg_manager.get_node("meaning:사랑_성장")
        self.assertIsNotNone(new_node)
        self.assertEqual(new_node['discovery_source'], 'Cell_Ascension_Ritual')
        self.assertEqual(new_node['type'], 'concept')
        self.assertEqual(new_node['parents'], ["사랑", "성장"])

        # Check that the hypothesis is cleaned up
        self.assertIsNone(self.pipeline.pending_hypothesis)
        self.assertEqual(len(self.core_memory.get_unasked_hypotheses()), 0)

if __name__ == '__main__':
    unittest.main()
