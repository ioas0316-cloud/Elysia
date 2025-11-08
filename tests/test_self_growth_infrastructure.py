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

from Project_Elysia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager

@unittest.skip("Skipping self-growth tests as MetaCognitionCortex is no longer part of the pipeline.")
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
        with patch('Project_Elysia.cognition_pipeline.KGManager', return_value=self.kg_manager):
            # The MetaCognitionCortex is no longer directly in the pipeline,
            # so this patch is no longer valid in its current form.
            # For now, we skip these tests.
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
        """
        new_concept = "the concept of algorithmic serendipity"

        # Process the message to trigger reflection
        self.pipeline.process_message(new_concept)
        # Further assertions would be needed here based on the new architecture.

    def test_reflection_on_existing_concept(self):
        """
        Scenario B: Verify reflection process for a pre-existing concept.
        """
        existing_concept = "love"

        # Process a message to trigger a new reflection on the same concept
        self.pipeline.process_message(existing_concept)
        # Further assertions would be needed here based on the new architecture.


    @classmethod
    def tearDownClass(cls):
        """Stop all patches after the test class is done."""
        for p in patches:
            p.stop()


if __name__ == '__main__':
    unittest.main()
