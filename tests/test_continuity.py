import unittest
import os
import json
from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core_memory import CoreMemory, Memory

class TestContinuityProtocol(unittest.TestCase):

    def setUp(self):
        """Set up a temporary state file for testing."""
        self.state_file_path = 'data/test_elysia_state.json'
        # Ensure the file does not exist before a test run
        if os.path.exists(self.state_file_path):
            os.remove(self.state_file_path)

    def tearDown(self):
        """Clean up the temporary state file after testing."""
        if os.path.exists(self.state_file_path):
            os.remove(self.state_file_path)

    def test_memory_persistence(self):
        """
        Tests if CoreMemory state is successfully saved and loaded across different
        CognitionPipeline instances.
        """
        # --- Phase 1: First session ---
        # Create a fresh CoreMemory and inject it
        initial_memory = CoreMemory(file_path=self.state_file_path)
        pipeline1 = CognitionPipeline(core_memory=initial_memory)

        # Interact with the pipeline to change the memory state
        test_memory_content = "This is a test experience from the first session."
        pipeline1.core_memory.add_experience(Memory(timestamp="2025-11-04T22:30:00Z", content=test_memory_content))
        pipeline1.core_memory.update_identity("test_user", {"last_seen": "2025-11-04T22:30:00Z"})

        # Manually save the state, simulating application shutdown
        pipeline1.core_memory._save_memory()

        # Verify that the state file was actually created
        self.assertTrue(os.path.exists(self.state_file_path))

        # --- Phase 2: Second session ---
        # Create a new CoreMemory, which should load from the file
        reloaded_memory = CoreMemory(file_path=self.state_file_path)
        pipeline2 = CognitionPipeline(core_memory=reloaded_memory)

        # --- Verification ---
        # 1. Check if the experience from the first session is present
        experiences = pipeline2.core_memory.get_experiences()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]['content'], test_memory_content)

        # 2. Check if the identity information from the first session is present
        identity = pipeline2.core_memory.get_identity()
        self.assertIn("test_user", identity)
        self.assertEqual(identity["test_user"]["last_seen"], "2025-11-04T22:30:00Z")

        print("\n[Continuity Test] SUCCESS: Memory was successfully persisted and reloaded.")

if __name__ == '__main__':
    unittest.main()
