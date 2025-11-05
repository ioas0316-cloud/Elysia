import unittest
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Sophia.core_memory import Memory, EmotionalState

class TestPipelineFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a fresh pipeline for each test."""
        self.pipeline = CognitionPipeline()
        self.pipeline.api_available = False

    def tearDown(self):
        """Clean up after each test."""
        pass

    def test_internal_response_ignores_conversational_memory(self):
        """
        Tests that the internal voice currently does not use conversational memory.
        """
        response, _ = self.pipeline.process_message("What do you know about black holes?")

        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)

    def test_internal_response_for_another_unknown_concept(self):
        """
        Tests the default response for another unknown concept to ensure consistency.
        """
        response, _ = self.pipeline.process_message("What is a supermassive black hole?")

        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)

    def test_internal_response_for_known_concept(self):
        """
        Tests that the internal voice can generate a response for a concept in the KG.
        """
        response, _ = self.pipeline.process_message("Tell me about photosynthesis.")

        prefix = "나는 지금 네 뜻을 더 선명히 이해하고자 해. "
        self.assertTrue(response['text'].startswith(prefix))

        # The stable reasoner will list all relationships.
        expected_facts = [
            "'light_source'은(는) 'photosynthesis'의 원인이 될 수 있습니다.",
            "'photosynthesis'은(는) '산소 발생'을(를) 유발할 수 있습니다.",
            "'photosynthesis'은(는) '식물 성장'을(를) 유발할 수 있습니다.",
            "'물'은(는) 'photosynthesis'와(과) 'supports' 관계를 가집니다."
        ]

        for fact in expected_facts:
            self.assertIn(fact, response['text'])

    def test_internal_response_for_unknown_concept(self):
        """
        Tests that the internal voice provides a default response for unknown concepts.
        """
        response, _ = self.pipeline.process_message("What is the weather like today?")

        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)


if __name__ == '__main__':
    unittest.main()
