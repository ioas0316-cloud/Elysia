import unittest
import os
import sys
from unittest.mock import MagicMock, patch, ANY

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class we are testing
from Project_Elysia.cognition_pipeline import CognitionPipeline
# Import classes to be mocked as dependencies
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.world import World
from Project_Sophia.emotional_engine import EmotionalEngine


class TestCentralDispatchPipeline(unittest.TestCase):

    def setUp(self):
        """Set up mock dependencies for the pipeline."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_cellular_world = MagicMock(spec=World)
        self.mock_emotional_engine = MagicMock(spec=EmotionalEngine)

        # Instantiate the pipeline for each test
        self.pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world,
            self.mock_emotional_engine
        )
        # Mock the handlers that are now members of the pipeline
        self.pipeline.hypothesis_handler = MagicMock()
        self.pipeline.command_handler = MagicMock()
        self.pipeline.default_reasoning_handler = MagicMock()

    def test_routes_to_hypothesis_handler_for_pending_hypothesis(self):
        """Verify routing to HypothesisHandler when a hypothesis is pending."""
        # 1. Setup context
        self.pipeline.conversation_context.pending_hypothesis = {'head': 'A', 'tail': 'B'}

        # 2. Process message
        self.pipeline.process_message("Yes, that makes sense.")

        # 3. Assert correct handler was called
        self.pipeline.hypothesis_handler.handle_response.assert_called_once()
        self.pipeline.command_handler.handle.assert_not_called()
        self.pipeline.default_reasoning_handler.handle.assert_not_called()

    def test_routes_to_command_handler_for_command_word(self):
        """Verify routing to CommandWordHandler for a command message."""
        # 1. Setup mock for can_handle
        self.pipeline.command_handler.can_handle.return_value = True

        # 2. Process message
        self.pipeline.process_message("계산: 2 + 2")

        # 3. Assert correct handler was called
        self.pipeline.command_handler.can_handle.assert_called_once_with("계산: 2 + 2")
        self.pipeline.command_handler.handle.assert_called_once()
        self.pipeline.hypothesis_handler.handle_response.assert_not_called()
        self.pipeline.default_reasoning_handler.handle.assert_not_called()

    def test_routes_to_hypothesis_handler_to_ask_new_question(self):
        """Verify routing to HypothesisHandler to ask a new hypothesis."""
        # 1. Setup mocks
        self.pipeline.conversation_context.pending_hypothesis = None
        self.pipeline.command_handler.can_handle.return_value = False
        self.pipeline.hypothesis_handler.should_ask_new_hypothesis.return_value = True

        # 2. Process message
        self.pipeline.process_message("What do you think about this?")

        # 3. Assert correct handler was called
        self.pipeline.hypothesis_handler.should_ask_new_hypothesis.assert_called_once()
        self.pipeline.hypothesis_handler.handle_ask.assert_called_once()
        self.pipeline.default_reasoning_handler.handle.assert_not_called()

    def test_routes_to_default_reasoning_handler_as_fallback(self):
        """Verify routing to DefaultReasoningHandler for a general message."""
        # 1. Setup mocks
        self.pipeline.conversation_context.pending_hypothesis = None
        self.pipeline.command_handler.can_handle.return_value = False
        self.pipeline.hypothesis_handler.should_ask_new_hypothesis.return_value = False

        # 2. Process message
        self.pipeline.process_message("Tell me about love.")

        # 3. Assert correct handler was called
        self.pipeline.default_reasoning_handler.handle.assert_called_once()
        self.pipeline.hypothesis_handler.handle_response.assert_not_called()
        self.pipeline.command_handler.handle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
