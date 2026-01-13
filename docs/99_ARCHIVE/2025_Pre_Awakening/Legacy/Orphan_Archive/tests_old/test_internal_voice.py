import unittest
from unittest.mock import MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.architecture.handlers import DefaultReasoningHandler
from Core.FoundationLayer.Foundation.emotional_engine import EmotionalEngine
from Core.FoundationLayer.Foundation.core.thought import Thought
from Project_Elysia.architecture.context import ConversationContext
from Core.FoundationLayer.Foundation.emotional_engine import EmotionalState
from Core.FoundationLayer.Foundation.logical_reasoner import LogicalReasoner
from Project_Elysia.value_centered_decision import ValueCenteredDecision
from Core.FoundationLayer.Foundation.insight_synthesizer import InsightSynthesizer
from Core.FoundationLayer.Foundation.response_styler import ResponseStyler
from Project_Mirror.creative_cortex import CreativeCortex
from Project_Mirror.perspective_cortex import PerspectiveCortex
from Core.FoundationLayer.Foundation.question_generator import QuestionGenerator
import logging

class TestDefaultReasoningHandler(unittest.TestCase):

    def setUp(self):
        """Set up mock components for the handler for each test."""
        self.mock_reasoner = MagicMock(spec=LogicalReasoner)
        self.mock_vcd = MagicMock(spec=ValueCenteredDecision)
        self.mock_synthesizer = MagicMock(spec=InsightSynthesizer)
        self.mock_styler = MagicMock(spec=ResponseStyler)
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.mock_creative_cortex = MagicMock(spec=CreativeCortex)
        self.mock_perspective_cortex = MagicMock(spec=PerspectiveCortex)
        self.mock_question_generator = MagicMock(spec=QuestionGenerator)
        self.mock_emotional_engine = MagicMock(spec=EmotionalEngine)

        # Instantiate the handler with all dependencies mocked
        self.handler = DefaultReasoningHandler(
            reasoner=self.mock_reasoner,
            vcd=self.mock_vcd,
            synthesizer=self.mock_synthesizer,
            creative_cortex=self.mock_creative_cortex,
            styler=self.mock_styler,
            logger=self.mock_logger,
            perspective_cortex=self.mock_perspective_cortex,
            question_generator=self.mock_question_generator,
            emotional_engine=self.mock_emotional_engine
        )

    def test_handler_orchestrates_thought_flow_correctly(self):
        """
        Tests the handler's main logic: deduce -> select -> synthesize -> style.
        """
        # 1. Configure mock behaviors
        thought = Thought(content="블랙홀은 중력과 관련이 있다.", source='kg', confidence=0.9)
        potential_thoughts = [thought]
        self.mock_reasoner.deduce_facts.return_value = potential_thoughts
        self.mock_vcd.select_thought.return_value = thought
        self.mock_synthesizer.synthesize.return_value = "블랙홀에 대한 합성된 생각."
        self.mock_styler.style_response.return_value = "[Styled] 블랙홀 생각."

        # 2. Call the handler
        context = ConversationContext()
        emotional_state = MagicMock(spec=EmotionalState)
        result = self.handler.handle("블랙홀에 대해 알려줘", context, emotional_state)

        # 3. Assert the interactions and final result
        self.mock_reasoner.deduce_facts.assert_called_once_with("블랙홀에 대해 알려줘")
        self.mock_vcd.select_thought.assert_called_once_with(
            candidates=potential_thoughts,
            context=["블랙홀에 대해 알려줘"],
            emotional_state=emotional_state,
            guiding_intention=None
        )
        self.mock_synthesizer.synthesize.assert_called_once_with([thought])
        self.mock_styler.style_response.assert_called_once_with("블랙홀에 대한 합성된 생각.", emotional_state)
        self.assertEqual(result['text'], "[Styled] 블랙홀 생각.")

    def test_handler_returns_default_response_when_no_thoughts_deduced(self):
        """
        Tests the handler's behavior when the reasoner returns no thoughts.
        """
        # 1. Configure mock behaviors
        self.mock_reasoner.deduce_facts.return_value = []
        self.mock_styler.style_response.return_value = "[Styled] 흥미로운 관점."

        # 2. Call the handler
        context = ConversationContext()
        emotional_state = MagicMock(spec=EmotionalState)
        result = self.handler.handle("알 수 없는 주제", context, emotional_state)

        # 3. Assert the interactions and final result
        self.mock_reasoner.deduce_facts.assert_called_once_with("알 수 없는 주제")
        self.mock_vcd.select_thought.assert_not_called()
        self.mock_synthesizer.synthesize.assert_not_called()
        self.mock_styler.style_response.assert_called_once_with("흥미로운 관점이네요. 조금 더 생각해볼게요.", emotional_state)
        self.assertEqual(result['text'], "[Styled] 흥미로운 관점.")


if __name__ == '__main__':
    unittest.main()
