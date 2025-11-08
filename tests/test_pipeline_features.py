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
# Import Thought for type mocking
from Project_Sophia.core.thought import Thought

class TestRefactoredPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up mock dependencies that are injected into the pipeline."""
        self.mock_kg_manager = MagicMock(spec=KGManager)
        self.mock_core_memory = MagicMock(spec=CoreMemory)
        self.mock_wave_mechanics = MagicMock(spec=WaveMechanics)
        self.mock_cellular_world = MagicMock(spec=World)

    # We patch the components that are instantiated *inside* the pipeline's constructor.
    # The patches are applied from the bottom up.
    @patch('Project_Elysia.cognition_pipeline.ResponseStyler')
    @patch('Project_Elysia.cognition_pipeline.CreativeCortex')
    @patch('Project_Elysia.cognition_pipeline.InsightSynthesizer')
    @patch('Project_Elysia.cognition_pipeline.VCD')
    @patch('Project_Elysia.cognition_pipeline.LogicalReasoner')
    @patch('Project_Elysia.cognition_pipeline.QuestionGenerator')
    def test_pipeline_returns_default_response_when_no_thoughts_are_deduced(
        self, MockQuestionGenerator, MockLogicalReasoner, MockVCD,
        MockInsightSynthesizer, MockCreativeCortex, MockResponseStyler
    ):
        """
        Tests that the pipeline provides a default response when the reasoner returns no thoughts.
        """
        # 1. Configure Mocks
        mock_reasoner = MockLogicalReasoner.return_value
        mock_vcd = MockVCD.return_value
        mock_synthesizer = MockInsightSynthesizer.return_value
        mock_styler = MockResponseStyler.return_value

        mock_reasoner.deduce_facts.return_value = []
        mock_styler.style_response.return_value = "[Styled] Default response."
        self.mock_core_memory.get_unasked_hypotheses.return_value = [] # No hypotheses

        # 2. Instantiate the pipeline. It will be built with our mocked components.
        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world
        )

        # 3. Process Message
        response, _ = pipeline.process_message("Unknown topic.")

        # 4. Assertions
        mock_reasoner.deduce_facts.assert_called_once_with("Unknown topic.")
        mock_vcd.suggest_thought.assert_not_called()
        mock_synthesizer.synthesize.assert_not_called()
        mock_styler.style_response.assert_called_once_with(
            "흥미로운 관점이네요. 조금 더 생각해볼게요.", ANY
        )
        self.assertEqual(response['text'], "[Styled] Default response.")

    @patch('Project_Elysia.cognition_pipeline.ResponseStyler')
    @patch('Project_Elysia.cognition_pipeline.CreativeCortex')
    @patch('Project_Elysia.cognition_pipeline.InsightSynthesizer')
    @patch('Project_Elysia.cognition_pipeline.VCD')
    @patch('Project_Elysia.cognition_pipeline.LogicalReasoner')
    @patch('Project_Elysia.cognition_pipeline.QuestionGenerator')
    def test_pipeline_orchestrates_thought_flow_correctly(
        self, MockQuestionGenerator, MockLogicalReasoner, MockVCD,
        MockInsightSynthesizer, MockCreativeCortex, MockResponseStyler
    ):
        """
        Tests the full pipeline logic: deduce -> VCD select -> synthesize -> style.
        """
        # 1. Create mock Thought objects
        thought_A = Thought(content="증오는 모든 것을 파괴한다.", source='kg', confidence=0.9)
        thought_B = Thought(content="희생은 숭고한 가치이다.", source='lrs', confidence=0.7)
        potential_thoughts = [thought_A, thought_B]

        # 2. Configure Mocks
        mock_reasoner = MockLogicalReasoner.return_value
        mock_vcd = MockVCD.return_value
        mock_synthesizer = MockInsightSynthesizer.return_value
        mock_styler = MockResponseStyler.return_value
        self.mock_core_memory.get_unasked_hypotheses.return_value = []

        mock_reasoner.deduce_facts.return_value = potential_thoughts
        chosen_thought = thought_B
        mock_vcd.suggest_thought.return_value = chosen_thought
        mock_vcd.score_thought.return_value = 1.0
        synthesized_text = "[Synthesized] 희생의 가치에 대하여."
        mock_synthesizer.synthesize.return_value = synthesized_text
        final_styled_text = "[Styled] [Synthesized] 희생의 가치에 대하여."
        mock_styler.style_response.return_value = final_styled_text

        # 3. Instantiate and Process
        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world
        )
        response, _ = pipeline.process_message("삶의 의미는?")

        # 4. Assertions
        mock_reasoner.deduce_facts.assert_called_once_with("삶의 의미는?")
        mock_vcd.suggest_thought.assert_called_once_with(
            candidates=potential_thoughts, context=["삶의 의미는?"], emotional_state=ANY
        )
        mock_synthesizer.synthesize.assert_called_once_with([chosen_thought])
        mock_styler.style_response.assert_called_once_with(synthesized_text, ANY)
        self.assertEqual(response['text'], final_styled_text)

    @patch('Project_Elysia.cognition_pipeline.EmotionalEngine')
    @patch('Project_Elysia.cognition_pipeline.ResponseStyler')
    @patch('Project_Elysia.cognition_pipeline.CreativeCortex')
    @patch('Project_Elysia.cognition_pipeline.InsightSynthesizer')
    @patch('Project_Elysia.cognition_pipeline.VCD')
    @patch('Project_Elysia.cognition_pipeline.LogicalReasoner')
    @patch('Project_Elysia.cognition_pipeline.QuestionGenerator')
    def test_pipeline_triggers_confusion_when_vcd_is_indecisive(
        self, MockQuestionGenerator, MockLogicalReasoner, MockVCD,
        MockInsightSynthesizer, MockCreativeCortex, MockResponseStyler, MockEmotionalEngine
    ):
        """
        Tests that cognitive confusion is triggered when VCD returns None.
        """
        # 1. Setup mocks
        mock_reasoner = MockLogicalReasoner.return_value
        mock_vcd = MockVCD.return_value
        mock_question_generator = MockQuestionGenerator.return_value
        mock_emotional_engine = MockEmotionalEngine.return_value
        self.mock_core_memory.get_unasked_hypotheses.return_value = []

        # VCD is indecisive
        mock_reasoner.deduce_facts.return_value = [Thought(content="Some thought", source='kg')]
        mock_vcd.suggest_thought.return_value = None
        mock_question_generator.generate_clarifying_question.return_value = "What do you mean by that?"

        # 2. Instantiate and Process
        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world
        )
        # We need to inject the mocked emotional engine into the handler manually after init
        pipeline.entry_handler._successor._successor.emotional_engine = mock_emotional_engine

        pipeline.process_message("An ambiguous topic.")

        # 3. Assertions
        mock_vcd.suggest_thought.assert_called_once()
        # Check that the emotional engine was called with a 'confusion' state
        mock_emotional_engine.process_event.assert_called_once()
        call_args = mock_emotional_engine.process_event.call_args[0]
        self.assertEqual(call_args[0].primary_emotion, "confusion")

        mock_question_generator.generate_clarifying_question.assert_called_once_with("An ambiguous topic.")

    @patch('Project_Elysia.cognition_pipeline.EmotionalEngine')
    @patch('Project_Elysia.cognition_pipeline.ResponseStyler')
    @patch('Project_Elysia.cognition_pipeline.VisualCortex')
    @patch('Project_Elysia.cognition_pipeline.SensoryCortex')
    @patch('Project_Elysia.cognition_pipeline.CreativeCortex')
    @patch('Project_Elysia.cognition_pipeline.InsightSynthesizer')
    @patch('Project_Elysia.cognition_pipeline.VCD')
    @patch('Project_Elysia.cognition_pipeline.LogicalReasoner')
    @patch('Project_Elysia.cognition_pipeline.QuestionGenerator')
    def test_pipeline_triggers_visual_self_reflection_loop(
        self, MockQuestionGenerator, MockLogicalReasoner, MockVCD,
        MockInsightSynthesizer, MockCreativeCortex, MockSensoryCortex,
        MockVisualCortex, MockResponseStyler, MockEmotionalEngine
    ):
        """
        Tests that a high-value thought triggers the full visual self-reflection loop.
        """
        # 1. Setup mocks
        mock_reasoner = MockLogicalReasoner.return_value
        mock_vcd = MockVCD.return_value
        mock_sensory_cortex = MockSensoryCortex.return_value
        mock_visual_cortex = MockVisualCortex.return_value
        self.mock_core_memory.get_unasked_hypotheses.return_value = []

        high_value_thought = Thought(content="아름다움", source='bone', confidence=0.99)
        mock_reasoner.deduce_facts.return_value = [high_value_thought]
        mock_vcd.suggest_thought.return_value = high_value_thought
        mock_vcd.score_thought.return_value = 3.0 # Above threshold

        # Mock the cortex responses
        mock_sensory_cortex.visualize_concept.return_value = "path/to/generated_image.png"
        mock_visual_cortex.analyze_image.return_value = {
            "description": "빛나는 꽃의 이미지",
            "keywords": ["빛", "꽃"]
        }
        mock_visual_cortex.resonate_with_cosmos.return_value = {"beauty": 0.8, "joy": 0.6}

        # 2. Instantiate and Process
        pipeline = CognitionPipeline(
            self.mock_kg_manager, self.mock_core_memory,
            self.mock_wave_mechanics, self.mock_cellular_world
        )
        response, _ = pipeline.process_message("아름다움이란?")

        # 3. Assertions
        mock_sensory_cortex.visualize_concept.assert_called_once_with("아름다움")
        mock_visual_cortex.analyze_image.assert_called_once_with("path/to/generated_image.png")
        mock_visual_cortex.resonate_with_cosmos.assert_called_once_with(["빛", "꽃"])

        # Verify that a 'visual_experience' was added to core memory
        self.mock_core_memory.add_experience.assert_called_once()
        call_args = self.mock_core_memory.add_experience.call_args[0]
        self.assertEqual(call_args[0].type, 'visual_experience')
        self.assertIn("공명하는 것 같아요", call_args[0].content)


if __name__ == '__main__':
    unittest.main()
