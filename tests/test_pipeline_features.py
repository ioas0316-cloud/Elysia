import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.core_memory import Memory, EmotionalState

class TestPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline for each test."""
        self.pipeline = CognitionPipeline()

    def tearDown(self):
        """Clean up after each test."""
        pass # No cleanup needed for now as tests are in-memory

    def test_internal_response_ignores_conversational_memory(self):
        """
        Tests that the internal voice currently does not use conversational memory.
        (Previously test_conversational_memory_is_retrieved)
        """
        # 1. Add a relevant memory to the core memory (optional, for context)
        past_experience = Memory(
            timestamp="2025-01-01T12:00:00",
            content="I enjoy learning about black holes.",
            emotional_state=EmotionalState(0.5, 0.5, 0.2, "curiosity", [])
        )
        # self.pipeline.core_memory.add_experience(past_experience) # Not needed for this test

        # 2. Ask a question about a topic that is not in the KG, but is in memory
        response, _ = self.pipeline.process_message("What do you know about black holes?")

        # 3. Assert that the response is the default "I don't know" message,
        # because the internal voice does not yet consult memory.
        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)

    def test_internal_response_for_another_unknown_concept(self):
        """
        Tests the default response for another unknown concept to ensure consistency.
        (Previously test_inquisitive_mind_is_triggered)
        """
        # Ask a question about a topic that is not in the memory or KG
        response, _ = self.pipeline.process_message("What is a supermassive black hole?")

        # Assert that the response is the generic learning message.
        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)

    def test_internal_response_for_known_concept(self):
        """
        Tests that the internal voice can generate a response for a concept in the KG.
        (Previously test_fallback_mechanism_on_api_key_error)
        """
        response, _ = self.pipeline.process_message("Tell me about photosynthesis.")

        # With the internal voice, the pipeline formulates a response based on the KG.
        # The order of facts is not guaranteed, so we check for presence of each fact.
        prefix = "나는 지금 네 뜻을 더 선명히 이해하고자 해. "
        self.assertTrue(response['text'].startswith(prefix))

        expected_facts = [
            "'light_source'은(는) 'photosynthesis'의 원인이 될 수 있습니다. (인과 강도: 0.9) (조건: chlorophyll)",
            "'photosynthesis'은(는) '산소 발생'을(를) 유발할 수 있습니다. (인과 강도: 0.87) (조건: 식물 성장)",
            "'photosynthesis'은(는) '식물 성장'을(를) 유발할 수 있습니다. (인과 강도: 0.85) (조건: 빛, 물)",
            "'photosynthesis'은(는) '물'와(과) 'supports' 관계를 가집니다."
        ]

        for fact in expected_facts:
            self.assertIn(fact, response['text'])

    def test_internal_response_for_unknown_concept(self):
        """
        Tests that the internal voice provides a default response for unknown concepts.
        (Previously test_fallback_mechanism_on_api_request_error)
        """
        response, _ = self.pipeline.process_message("What is the weather like today?")

        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)


if __name__ == '__main__':
    unittest.main()
