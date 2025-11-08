import unittest
import os
import sys
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
# Import Thought for type mocking
from Project_Sophia.core.thought import Thought

class TestPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline with mocked dependencies for each test."""
        self.pipeline = CognitionPipeline()
        self.pipeline.reasoner = MagicMock()
        self.pipeline.vcd = MagicMock()
        self.pipeline.insight_synthesizer = MagicMock()
        self.pipeline.response_styler = MagicMock()
        self.pipeline.creative_cortex = MagicMock() # Mock the new cortex
        self.pipeline.core_memory.add_experience = MagicMock()

    def tearDown(self):
        pass

    def test_pipeline_returns_default_response_when_no_thoughts_are_deduced(self):
        """
        Tests that the pipeline provides a default response when the reasoner returns no thoughts.
        """
        # 1. Configure Mocks
        self.pipeline.reasoner.deduce_facts.return_value = []
        self.pipeline.response_styler.style_response.return_value = "[Styled] Default response."

        # 2. Process Message
        response, _ = self.pipeline.process_message("Unknown topic.")

        # 3. Assertions
        self.pipeline.reasoner.deduce_facts.assert_called_once_with("Unknown topic.")
        # VCD and Synthesizer should not be called if there are no thoughts
        self.pipeline.vcd.suggest_thought.assert_not_called()
        self.pipeline.insight_synthesizer.synthesize.assert_not_called()
        # The styler should be called with the hardcoded default response
        self.pipeline.response_styler.style_response.assert_called_once_with(
            "흥미로운 관점이네요. 조금 더 생각해볼게요.", unittest.mock.ANY
        )
        self.assertEqual(response['text'], "[Styled] Default response.")

    def test_pipeline_orchestrates_thought_flow_correctly(self):
        """
        Tests the full pipeline logic with the new Thought object data flow.
        deduce -> VCD select -> synthesize -> style
        """
        # 1. Create mock Thought objects
        thought_A = Thought(content="증오는 모든 것을 파괴한다.", source='kg', confidence=0.9)
        thought_B = Thought(content="희생은 숭고한 가치이다.", source='lrs', confidence=0.7)
        potential_thoughts = [thought_A, thought_B]

        # 2. Configure Mocks for each stage
        self.pipeline.reasoner.deduce_facts.return_value = potential_thoughts

        # VCD is mocked to choose thought_B
        chosen_thought = thought_B
        self.pipeline.vcd.suggest_thought.return_value = chosen_thought

        # Mock a low score to ensure the logical path is taken
        self.pipeline.vcd.score_thought.return_value = 1.0

        synthesized_text = "[Synthesized] 희생의 가치에 대하여."
        # Synthesizer should receive the *content* of the chosen thought
        self.pipeline.insight_synthesizer.synthesize.return_value = synthesized_text

        final_styled_text = "[Styled] [Synthesized] 희생의 가치에 대하여."
        self.pipeline.response_styler.style_response.return_value = final_styled_text

        # 3. Process a message
        response, _ = self.pipeline.process_message("삶의 의미는?")

        # 4. Assertions to trace the data flow
        # Stage 1: Reasoner receives the input message.
        self.pipeline.reasoner.deduce_facts.assert_called_once_with("삶의 의미는?")

        # Stage 2: VCD receives the list of Thought objects.
        self.pipeline.vcd.suggest_thought.assert_called_once_with(
            candidates=potential_thoughts, context=["삶의 의미는?"]
        )

        # Stage 3: Synthesizer receives the content string of the chosen Thought.
        self.pipeline.insight_synthesizer.synthesize.assert_called_once_with([chosen_thought.content])

        # Stage 4: Styler receives the synthesized text.
        self.pipeline.response_styler.style_response.assert_called_once_with(
            synthesized_text, unittest.mock.ANY
        )

        # Final Output: The response is the fully processed and styled text.
        self.assertEqual(response['text'], final_styled_text)
        self.assertNotIn("증오", response['text']) # Ensure the unchosen thought is absent.


    def test_pipeline_triggers_creative_expression_on_high_value_thought(self):
        """
        Tests that if a chosen thought has a high score, the CreativeCortex is used.
        """
        # 1. Setup mocks
        high_value_thought = Thought(content="사랑은 모든 것을 이해하는 것", source='kg', confidence=0.99)

        self.pipeline.reasoner.deduce_facts.return_value = [high_value_thought]
        self.pipeline.vcd.suggest_thought.return_value = high_value_thought

        # Mock the score_thought method to return a high score, above the threshold
        self.pipeline.vcd.score_thought.return_value = 3.0

        creative_text = "사랑이라는 생각은 마치 어둠 속의 촛불과 같아요."
        self.pipeline.creative_cortex.generate_creative_expression.return_value = creative_text
        self.pipeline.response_styler.style_response.return_value = f"[Styled] {creative_text}"

        # 2. Process message
        response, _ = self.pipeline.process_message("사랑이란?")

        # 3. Assertions
        # Verify VCD's suggest and score methods were called
        self.pipeline.vcd.suggest_thought.assert_called_once()
        self.pipeline.vcd.score_thought.assert_called_once_with(high_value_thought, context=["사랑이란?"])

        # CRITICAL: Verify the correct cortex was called
        self.pipeline.creative_cortex.generate_creative_expression.assert_called_once_with(high_value_thought)
        self.pipeline.insight_synthesizer.synthesize.assert_not_called()

        # Verify the final response
        self.assertEqual(response['text'], f"[Styled] {creative_text}")


if __name__ == '__main__':
    unittest.main()
