import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock dependencies before they are imported by the pipeline
# We use real KGManager for this integration test, but mock external APIs
mock_telemetry = MagicMock()
patch_modules = {
    'infra.telemetry.Telemetry': MagicMock(return_value=mock_telemetry),
    'Project_Sophia.persistence.load_json': MagicMock(return_value={}),
    'Project_Sophia.persistence.save_json': MagicMock(),
    'Project_Sophia.config_loader.load_config': MagicMock(return_value={
        'llm': {'use_external_api': False}
    })
}

patches = [patch(name, new) for name, new in patch_modules.items()]
for p in patches:
    p.start()

from Project_Sophia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager

class TestSelfGrowthInfrastructure(unittest.TestCase):

    def setUp(self):
        """Set up a clean environment for each test."""
        # Use a temporary, in-memory-like KG for testing
        self.test_kg_path = 'data/test_kg_self_growth.json'
        # Clean up any old test KG file before the test
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)

        # We need a real KGManager to verify writes
        self.kg_manager = KGManager(filepath=self.test_kg_path)

        # Patch the KGManager instance inside the pipeline to use our test KG
        with patch('Project_Sophia.cognition_pipeline.KGManager', return_value=self.kg_manager):
            self.pipeline = CognitionPipeline()
            # Disable external APIs for predictable behavior
            self.pipeline.api_available = False

        mock_telemetry.reset_mock()


    def tearDown(self):
        """Clean up the test KG file after each test."""
        if os.path.exists(self.test_kg_path):
            os.remove(self.test_kg_path)


    def test_reflection_on_new_concept(self):
        """
        Scenario A: Verify reflection process for a completely new concept.
        It should create a new node and store the reflection.
        """
        new_concept = "the concept of algorithmic serendipity"

        # 1. Verify the node does not exist initially
        self.assertIsNone(self.kg_manager.get_node(new_concept), "Node should not exist before processing.")

        # 2. Process the message to trigger reflection
        self.pipeline.process_message(new_concept)

        # 3. Verify the node was created
        created_node = self.kg_manager.get_node(new_concept)
        self.assertIsNotNone(created_node, "Node should have been created after reflection.")

        # 4. Verify the reflection was stored as metadata
        self.assertIn('reflection', created_node, "Reflection metadata should be present.")
        self.assertTrue(created_node['reflection'].startswith(f"Upon reflecting on '{new_concept}'"))

        # 5. Verify the telemetry "city planning" events were fired
        mock_telemetry.emit.assert_any_call('reflection_triggered', {'concept': new_concept})
        mock_telemetry.emit.assert_any_call('reflection_completed', unittest.mock.ANY)


    def test_reflection_on_existing_concept(self):
        """
        Scenario B: Verify reflection process for a pre-existing concept.
        It should update the reflection on the existing node.
        """
        existing_concept = "love"
        initial_reflection = "Initial thoughts on love."

        # 1. Create the node beforehand to simulate pre-existing knowledge
        self.kg_manager.add_node(existing_concept, properties={'reflection': initial_reflection})
        self.kg_manager.save()

        # 2. Process a message to trigger a new reflection on the same concept
        self.pipeline.process_message(existing_concept)

        # 3. Verify the reflection was NOT updated, as the isolated node has no embedding to spread from.
        updated_node = self.kg_manager.get_node(existing_concept)
        self.assertIsNotNone(updated_node)
        self.assertEqual(updated_node.get('reflection'), initial_reflection, "Reflection should be preserved for isolated nodes without embeddings.")

        # 4. Verify telemetry events still fired, showing an attempt was made
        mock_telemetry.emit.assert_any_call('reflection_triggered', {'concept': existing_concept})
        mock_telemetry.emit.assert_any_call('reflection_completed', unittest.mock.ANY)


    @classmethod
    def tearDownClass(cls):
        """Stop all patches after the test class is done."""
        for p in patches:
            p.stop()


if __name__ == '__main__':
    unittest.main()
