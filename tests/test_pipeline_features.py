import unittest
import os
import sys
from unittest.mock import MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory import Memory, EmotionalState

class TestPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline for each test."""
        self.pipeline = CognitionPipeline()
        # Mock all dependencies to isolate the pipeline's orchestration logic
        self.pipeline.reasoner = MagicMock()
        self.pipeline.vcd = MagicMock()
        self.pipeline.insight_synthesizer = MagicMock()
        self.pipeline.response_styler = MagicMock()
        # Prevent file I/O during tests
        self.pipeline.core_memory.add_experience = MagicMock()

    def tearDown(self):
        """Clean up after each test."""
        pass

    def test_pipeline_returns_default_response_when_no_facts_are_deduced(self):
        """
        Tests the new default response path when the reasoner finds no facts.
        """
        # 1. Configure mocks
        self.pipeline.reasoner.deduce_facts.return_value = []
        # The styler will receive the default text and format it.
        self.pipeline.response_styler.style_response.return_value = "[Styled] 흥미로운 관점이네요. 조금 더 생각해볼게요."

        # 2. Process a message
        response, _ = self.pipeline.process_message("Tell me about something unknown.")

        # 3. Assertions
        # Verify the reasoner was called.
        self.pipeline.reasoner.deduce_facts.assert_called_once_with("Tell me about something unknown.")
        # Verify that VCD and synthesizer were NOT called.
        self.pipeline.vcd.suggest_action.assert_not_called()
        self.pipeline.insight_synthesizer.synthesize.assert_not_called()
        # Verify the styler was called with the correct default text.
        self.pipeline.response_styler.style_response.assert_called_once_with(
            "흥미로운 관점이네요. 조금 더 생각해볼게요.", unittest.mock.ANY
        )
        # Assert the final response is the styled default message.
        self.assertEqual(response['text'], "[Styled] 흥미로운 관점이네요. 조금 더 생각해볼게요.")

    def test_pipeline_selects_most_value_aligned_response(self):
        """
        Tests the full pipeline logic: deduce -> VCD select -> synthesize -> style.
        """
        # 1. Configure mocks for each stage of the pipeline
        potential_facts = ["증오는 모든 것을 파괴한다.", "희생은 숭고한 가치이다."]
        self.pipeline.reasoner.deduce_facts.return_value = potential_facts

        chosen_fact = "희생은 숭고한 가치이다."
        self.pipeline.vcd.suggest_action.return_value = chosen_fact

        synthesized_text = "[Synthesized] 희생은 숭고한 가치입니다."
        self.pipeline.insight_synthesizer.synthesize.return_value = synthesized_text

        final_styled_text = "[Styled] [Synthesized] 희생은 숭고한 가치입니다."
        self.pipeline.response_styler.style_response.return_value = final_styled_text

        # 2. Process a message to trigger the full pipeline.
        response, _ = self.pipeline.process_message("삶의 의미는 무엇인가?")

        # 3. Assertions to trace the data flow
        # Stage 1: Reasoner gets the input message.
        self.pipeline.reasoner.deduce_facts.assert_called_once_with("삶의 의미는 무엇인가?")

        # Stage 2: VCD gets the candidates from the reasoner.
        self.pipeline.vcd.suggest_action.assert_called_once_with(
            candidates=potential_facts, context=["삶의 의미는 무엇인가?"]
        )

        # Stage 3: Synthesizer gets the single chosen fact from VCD.
        self.pipeline.insight_synthesizer.synthesize.assert_called_once_with([chosen_fact])

        # Stage 4: Styler gets the synthesized text from the synthesizer.
        self.pipeline.response_styler.style_response.assert_called_once_with(
            synthesized_text, unittest.mock.ANY
        )

        # Final Output: The response is the fully processed and styled text.
        self.assertEqual(response['text'], final_styled_text)
        # And crucially, the unchosen fact is not present.
        self.assertNotIn("증오", response['text'])


if __name__ == '__main__':
    unittest.main()
