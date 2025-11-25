
import sys
import os
import logging
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.cognition_pipeline import CognitionPipeline

class TestPlanningIntegration(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_kg_manager = MagicMock()
        self.mock_core_memory = MagicMock()
        self.mock_wave_mechanics = MagicMock()
        self.mock_cellular_world = MagicMock()
        self.mock_emotional_engine = MagicMock()
        self.mock_logger = logging.getLogger("TestLogger")
        self.mock_logger.setLevel(logging.INFO)
        
        # Initialize pipeline
        self.pipeline = CognitionPipeline(
            kg_manager=self.mock_kg_manager,
            core_memory=self.mock_core_memory,
            wave_mechanics=self.mock_wave_mechanics,
            cellular_world=self.mock_cellular_world,
            emotional_engine=self.mock_emotional_engine,
            logger=self.mock_logger
        )
        
        # Mock the PlanningCortex inside the pipeline to avoid real API calls
        self.pipeline.planning_cortex = MagicMock()
        self.pipeline.planning_handler.planning_cortex = self.pipeline.planning_cortex
        
        # Setup mock return value for develop_plan
        self.pipeline.planning_cortex.develop_plan.return_value = [
            {"tool_name": "mock_tool", "parameters": {"arg": "value"}}
        ]

    def test_planning_trigger(self):
        message = "Plan: Create a test file"
        
        # Process message
        result, _ = self.pipeline.process_message(message)
        
        # Verify that develop_plan was called
        self.pipeline.planning_cortex.develop_plan.assert_called_with("Create a test file")
        
        # Verify result contains the plan
        self.assertIn("plan", result)
        self.assertEqual(result["plan"][0]["tool_name"], "mock_tool")
        print("SUCCESS: Planning trigger verified.")

if __name__ == "__main__":
    unittest.main()
