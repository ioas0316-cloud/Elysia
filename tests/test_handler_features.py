import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class we are testing and its dependencies
from Project_Elysia.architecture.handlers import DefaultReasoningHandler
from Project_Sophia.core.thought import Thought
from Project_Elysia.architecture.context import ConversationContext
from Project_Sophia.emotional_engine import EmotionalState

class TestDefaultReasoningHandlerFeatures(unittest.TestCase):

    def setUp(self):
        """Set up mock dependencies for the DefaultReasoningHandler."""
        self.mock_reasoner = MagicMock()
        self.mock_vcd = MagicMock()
        self.mock_synthesizer = MagicMock()
        self.mock_creative_cortex = MagicMock()
        self.mock_styler = MagicMock()
        self.mock_logger = MagicMock()
        self.mock_perspective_cortex = MagicMock()
        self.mock_question_generator = MagicMock()
        self.mock_emotional_engine = MagicMock()

        # Instantiate the handler with mocked components
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

        # Set up default return values for mocks to avoid None errors
        self.mock_styler.style_response.return_value = "Styled response"
        self.mock_synthesizer.synthesize.return_value = "Synthesized insight"
        self.mock_creative_cortex.generate_creative_expression.return_value = "A creative poem."

    def test_creative_cortex_triggered_on_high_score_thought(self):
        """Verify CreativeCortex is called when a thought's VCD score is high."""
        # 1. Setup: Create a thought with a high score
        high_score_thought = Thought(content="This is a profound idea.", source="test")
        setattr(high_score_thought, 'vcd_score', 3.0) # Manually set the score

        # Mock the VCD to return this thought
        self.mock_vcd.select_thought.return_value = high_score_thought
        # Mock the reasoner to return a list containing this thought
        self.mock_reasoner.deduce_facts.return_value = [high_score_thought]

        # 2. Process message
        context = ConversationContext()
        emotional_state = EmotionalState(0.5, 0.5, 0.5, "neutral", [])
        result = self.handler.handle("A deep question", context, emotional_state)

        # 3. Assertions
        self.mock_creative_cortex.generate_creative_expression.assert_called_once_with(high_score_thought)
        self.assertEqual(result['type'], 'composite_insight')
        self.assertIn('creative_output', result)
        self.assertEqual(result['creative_output'], "A creative poem.")

    def test_creative_cortex_not_triggered_on_low_score_thought(self):
        """Verify CreativeCortex is NOT called when a thought's VCD score is low."""
        # 1. Setup: Create a thought with a low score
        low_score_thought = Thought(content="This is a simple fact.", source="test")
        setattr(low_score_thought, 'vcd_score', 1.0) # Manually set the score

        # Mock the VCD to return this thought
        self.mock_vcd.select_thought.return_value = low_score_thought
        # Mock the reasoner to return a list containing this thought
        self.mock_reasoner.deduce_facts.return_value = [low_score_thought]

        # 2. Process message
        context = ConversationContext()
        emotional_state = EmotionalState(0.5, 0.5, 0.5, "neutral", [])
        result = self.handler.handle("A simple question", context, emotional_state)

        # 3. Assertions
        self.mock_creative_cortex.generate_creative_expression.assert_not_called()
        self.assertEqual(result['type'], 'text')
        self.assertNotIn('creative_output', result)

if __name__ == '__main__':
    unittest.main()
