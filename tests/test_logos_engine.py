import unittest
from unittest.mock import Mock, patch

# HACK: Add project root to sys.path to allow absolute imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.meta_cognition_cortex import MetaCognitionCortex
from Project_Elysia.guardian import Guardian
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory import CoreMemory
from Project_Sophia.core.thought import Thought
from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics

class TestLogosEngine(unittest.TestCase):

    def setUp(self):
        """
        Set up a clean environment for each test.
        Crucially, each test gets its own instances of stateful objects like CoreMemory.
        """
        self.mock_logger = Mock()
        self.mock_kg_manager = Mock(spec=KGManager)
        self.mock_wave_mechanics = Mock(spec=WaveMechanics)

        # Mock the behavior of WaveMechanics for unit tests
        self.mock_wave_mechanics.get_resonance_between.return_value = 0.8

        # For unit tests, we'll mock CoreMemory's methods.
        self.mock_core_memory = Mock(spec=CoreMemory)
        self.mock_core_memory.get_identity.return_value = {'name': 'Elysia_Test'}
        self.mock_core_memory.get_experiences.return_value = []
        self.mock_core_memory.get_unasked_hypotheses.return_value = []


    def test_meditate_on_logos_unit(self):
        """
        Unit Test: Verify that meditate_on_logos generates a meaningful 'Intention Thought'.
        """
        # Arrange
        cortex = MetaCognitionCortex(self.mock_kg_manager, self.mock_wave_mechanics, self.mock_logger)

        # Act
        intention_thought = cortex.meditate_on_logos(self.mock_core_memory)

        # Assert
        self.assertIsInstance(intention_thought, Thought)
        self.assertEqual(intention_thought.source, 'logos_engine')
        self.assertIn("엘리시아의 명상", intention_thought.content)
        self.assertIn("Elysia_Test", intention_thought.content)
        self.assertEqual(intention_thought.evidence, ['사랑']) # Default focus
        self.mock_wave_mechanics.get_resonance_between.assert_called_with('사랑', 'love')


    def test_meditate_on_logos_with_hypothesis(self):
        """
        Unit Test: Verify that meditation focuses on an unasked hypothesis if present.
        """
        # Arrange
        cortex = MetaCognitionCortex(self.mock_kg_manager, self.mock_wave_mechanics, self.mock_logger)
        hypothesis = [{'head': '연결', 'tail': '성장', 'relation': 'enables'}]
        self.mock_core_memory.get_unasked_hypotheses.return_value = hypothesis

        # Act
        intention_thought = cortex.meditate_on_logos(self.mock_core_memory)

        # Assert
        self.assertEqual(intention_thought.evidence, ['연결'])
        self.assertIn("확인되지 않은 통찰('연결')", intention_thought.content)
        self.mock_wave_mechanics.get_resonance_between.assert_called_with('연결', 'love')


    @patch('Project_Elysia.cognition_pipeline.LogicalReasoner')
    def test_intention_affects_pipeline_integration(self, MockLogicalReasoner):
        """
        Integration Test: Verify that the guiding intention overrides other scores in VCD.
        """
        # 1. ARRANGE
        # Mock Reasoner to return two thoughts
        thought_A = Thought(content="우주에 대한 생각", source='bone', confidence=0.9, evidence=['우주'])
        thought_B = Thought(content="연결의 중요성", source='flesh', confidence=0.7, evidence=['연결'])
        mock_reasoner_instance = MockLogicalReasoner.return_value
        mock_reasoner_instance.deduce_facts.return_value = [thought_A, thought_B]

        # Use a real CoreMemory instance to track state changes
        # Use an in-memory test file by passing None
        real_core_memory = CoreMemory(file_path=None)
        real_core_memory.add_notable_hypothesis({'head': '연결', 'tail': '성장'})

        # Setup real components for the pipeline
        real_kg_manager = KGManager(filepath="tests/test_data/kg.json") # Use a test KG
        real_wave_mechanics = WaveMechanics(real_kg_manager)
        real_meta_cortex = MetaCognitionCortex(real_kg_manager, real_wave_mechanics, self.mock_logger)

        # The Guardian's role is to run meditation and set the intention
        # We simulate this part instead of running the full Guardian loop
        intention = real_meta_cortex.meditate_on_logos(real_core_memory)
        real_core_memory.add_guiding_intention(intention)
        self.assertEqual(real_core_memory.get_guiding_intention().evidence, ['연결'])

        # Create the pipeline with a mix of real and mocked components
        pipeline = CognitionPipeline(
            kg_manager=real_kg_manager,
            core_memory=real_core_memory,
            wave_mechanics=real_wave_mechanics,
            cellular_world=None, # Not needed for this test
            logger=self.mock_logger
        )
        # We need to replace the reasoner inside the pipeline's handler with our mock
        pipeline.entry_handler._successor._successor.reasoner = mock_reasoner_instance

        # 2. ACT
        result, _ = pipeline.process_message("어떻게 생각해?")

        # 3. ASSERT
        # Even though thought_A ('우주') from 'bone' would have a higher score
        # due to the wisdom bonus, the pipeline MUST choose thought_B ('연결')
        # because the guiding intention from the meditation cycle was '연결'.
        self.assertIn("연결의 중요성", result['text'])
        self.assertNotIn("우주", result['text'])
        # Correct the log message to match the f-string in VCD (`...` is always appended)
        self.mock_logger.info.assert_any_call("Intention Match: Prioritizing thought '연결의 중요성...' due to alignment with '연결'.")


if __name__ == '__main__':
    unittest.main()
