# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the main orchestrator and the cortex to be tested
from Project_Elysia.guardian import Guardian
from Project_Sophia.meta_cognition_cortex import MetaCognitionCortex

class TestMetaCognitionIntegration(unittest.TestCase):

    @patch('Project_Elysia.guardian.WebSearchCortex')
    @patch('Project_Elysia.guardian.KnowledgeDistiller')
    @patch('Project_Elysia.guardian.ExplorationCortex')
    @patch('Project_Elysia.elysia_daemon.ElysiaDaemon.run_cycle') # We don't need the daemon to run its own cycle
    @patch('Project_Sophia.meta_cognition_cortex.MetaCognitionCortex.log_event')
    def test_pipeline_event_is_captured_by_metacognition(
        self, mock_log_event, MockRunCycle, MockExplorationCortex,
        MockKnowledgeDistiller, MockWebSearchCortex
    ):
        """
        Tests that an event published by the CognitionPipeline is successfully
        received by the MetaCognitionCortex.
        """
        # 1. Instantiate the Guardian, which will set up the whole object graph.
        # This includes creating the daemon and the pipeline, and subscribing the
        # meta_cognition_cortex to the pipeline's event bus.
        guardian = Guardian()

        # 2. Get a direct reference to the pipeline from the guardian's daemon.
        pipeline = guardian.daemon.cognition_pipeline

        # 3. Process a simple message through the pipeline.
        # This should cause the pipeline to publish a "message_processed" event.
        pipeline.process_message("Hello Elysia")

        # 4. Assert that the log_event method on the MetaCognitionCortex instance
        #    was called with the correct event type and data.
        mock_log_event.assert_called_with(
            "message_processed",
            {"result": ANY}
        )

        # Verify it was called exactly once for this interaction
        self.assertEqual(mock_log_event.call_count, 1)

if __name__ == '__main__':
    unittest.main()