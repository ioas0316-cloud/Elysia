import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Main class to test
from Project_Elysia.guardian import Guardian
# Dependencies for the test
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World
from Project_Sophia.core.cell import Cell
from Project_Elysia.elysia_daemon import ElysiaDaemon
from Project_Elysia.guardian import PRIMORDIAL_DNA

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

        # --- Instantiate Core Components for Integration Test ---
        self.kg_manager = KGManager(filepath=self.test_kg_path)
        self.core_memory = CoreMemory(file_path=self.test_memory_path)
        self.wave_mechanics = WaveMechanics(self.kg_manager)
        self.cellular_world = World(primordial_dna=PRIMORDIAL_DNA)

        # --- Instantiate a real Guardian to test its hypothesis generation ---
        with patch('Project_Elysia.guardian.WebSearchCortex'), \
             patch('Project_Elysia.guardian.KnowledgeDistiller'), \
             patch('Project_Elysia.guardian.ExplorationCortex'):
            self.guardian = Guardian()
            # Inject our test components into the Guardian
            self.guardian.kg_manager = self.kg_manager
            self.guardian.core_memory = self.core_memory

        # --- Instantiate a real daemon and get the pipeline from it ---
        from Project_Sophia.meta_cognition_cortex import MetaCognitionCortex
        mock_meta_cortex = MagicMock(spec=MetaCognitionCortex)

        daemon = ElysiaDaemon(
            kg_manager=self.kg_manager,
            core_memory=self.core_memory,
            wave_mechanics=self.wave_mechanics,
            cellular_world=self.cellular_world,
            meta_cognition_cortex=mock_meta_cortex,
            logger=MagicMock()
        )
        self.pipeline = daemon.cognition_pipeline

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def test_full_ascension_process(self):
        """
        Verify the entire Cell Ascension process with the new architecture:
        1. Guardian identifies a 'meaning' cell and creates a hypothesis.
        2. Pipeline, on first message, asks the user for approval.
        3. Pipeline, on second message, processes approval and updates the KG.
        """
        # --- Step 1: A 'meaning' cell is born and Guardian creates a hypothesis ---
        ascension_candidate_cell = Cell(
            concept_id="meaning:사랑_성장",
            dna=PRIMORDIAL_DNA,
            initial_properties={"parents": ["사랑", "성장"]}
        )
        self.guardian._handle_ascension_candidates([ascension_candidate_cell])

        # Verify that the ascension hypothesis was created in Core Memory
        hypotheses = self.core_memory.get_unasked_hypotheses()
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0]['relation'], '승천')
        self.assertEqual(hypotheses[0]['head'], 'meaning:사랑_성장')

        # --- Step 2: The pipeline asks for approval ---
        response, _ = self.pipeline.process_message("아무 말") # Any message triggers the check
        self.assertIn("새로운 의미 'meaning:사랑_성장'가 탄생했습니다.", response['text'])
        self.assertIn("영원한 '개념'으로 승천시켜 지식의 일부로 만들까요?", response['text'])

        # Verify that a hypothesis is now pending in the pipeline's context
        self.assertIsNotNone(self.pipeline.conversation_context.pending_hypothesis)
        self.assertEqual(self.pipeline.conversation_context.pending_hypothesis['head'], 'meaning:사랑_성장')

        # --- Step 3: The user approves, and the node is created ---
        approval_message = "응, 승인할게"
        response, _ = self.pipeline.process_message(approval_message)
        self.assertIn("새로운 개념 'meaning:사랑_성장'이(가) 지식의 일부로 승천했습니다.", response['text'])

        # --- Step 4: Final Verification ---
        # Check that the new node exists in the Knowledge Graph
        new_node = self.kg_manager.get_node("meaning:사랑_성장")
        self.assertIsNotNone(new_node)
        self.assertEqual(new_node['discovery_source'], 'Cell_Ascension_Ritual')
        self.assertEqual(new_node['parents'], ["사랑", "성장"])

        # Check that the hypothesis is cleaned up from the context and memory
        self.assertIsNone(self.pipeline.conversation_context.pending_hypothesis)
        self.assertEqual(len(self.core_memory.get_unasked_hypotheses()), 0)

if __name__ == '__main__':
    unittest.main()
