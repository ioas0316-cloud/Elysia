import unittest
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to path to allow direct imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from Project_Sophia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager

class TestGrowthProtocol(unittest.TestCase):

    def setUp(self):
        """Set up a temporary knowledge graph and pipeline for each test."""
        self.test_kg_path = 'data/test_kg.json'
        self.test_memory_path = 'Elysia_Input_Sanctum/test_memory.json'

        # Ensure clean slate
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

        # Create a KG with a known fact
        self.kg_manager = KGManager(locked=False) # Start unlocked for setup
        self.kg_manager.kg = {"nodes": [], "edges": []} # Clear
        self.kg_manager.add_edge("socrates", "human", "is_a")
        self.kg_manager.save_to(self.test_kg_path)
        self.kg_manager.lock()

        # Patch the CognitionPipeline to use the test KG
        self.pipeline = CognitionPipeline()
        self.pipeline.kg_manager.kg = self.kg_manager.kg # Use the same kg instance
        self.pipeline.core_memory.file_path = self.test_memory_path # Use test memory

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def test_full_growth_cycle(self):
        # 1. Input a contradicting statement
        contradicting_input = "Socrates is not a human"
        response, _, _ = self.pipeline.process_message(contradicting_input)

        # 2. Verify it asks for permission
        self.assertIn("제 지식에 혼란이 생겼습니다", response)
        self.assertIn("수정하도록 허락해주시겠습니까?", response)

        # 3. Verify the pending action was saved in memory
        pending_actions = self.pipeline.core_memory.get_pending_actions()
        self.assertEqual(len(pending_actions), 1)
        self.assertEqual(pending_actions[0]['type'], 'remove_edge')
        self.assertEqual(pending_actions[0]['source'], 'socrates')

        # Reset pending actions since get_pending_actions clears them
        self.pipeline.core_memory.add_pending_action(pending_actions[0])

        # 4. Grant permission
        approval_input = "허락한다"
        response, _, context = self.pipeline.process_message(approval_input)

        # 5. Verify it acknowledges the permission
        self.assertIn("감사합니다, 창조자님", response)

        # 6. Simulate the daemon's execution logic
        actions_to_execute = context.get("execute_actions")
        self.assertIsNotNone(actions_to_execute)
        self.assertEqual(len(actions_to_execute), 1)

        # 7. Execute the growth moment
        self.pipeline.kg_manager.unlock()
        action = actions_to_execute[0]
        self.pipeline.kg_manager.remove_edge(action['source'], action['target'], action['relation'])
        self.pipeline.kg_manager.lock()

        # 8. Verify the knowledge was actually removed
        edge = self.pipeline.kg_manager.get_edge("socrates", "human", "is_a")
        self.assertIsNone(edge, "The conflicting edge should have been removed.")

if __name__ == '__main__':
    unittest.main()
