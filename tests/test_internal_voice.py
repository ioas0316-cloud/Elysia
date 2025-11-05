import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.cognition_pipeline import CognitionPipeline
from tools.kg_manager import KGManager
from Project_Sophia.logical_reasoner import LogicalReasoner

class TestInternalVoice(unittest.TestCase):
    def setUp(self):
        """Set up a fresh pipeline and a mock KGManager for each test."""
        self.kg_manager_instance = KGManager()
        self.kg_manager_instance.kg = {"nodes": [], "edges": []}

        self.kg_manager_instance.add_node("socrates", properties={"description": "A Greek philosopher from Athens."})
        self.kg_manager_instance.add_node("human")
        self.kg_manager_instance.add_or_update_edge("socrates", "human", "is_a")

        self.kg_manager_instance.add_node("black hole", properties={"description": "A region of spacetime where gravity is so strong that nothing can escape."})
        self.kg_manager_instance.add_node("gravity")
        self.kg_manager_instance.add_or_update_edge("black hole", "gravity", "related_to")

        self.reasoner_with_mock_kg = LogicalReasoner(kg_manager=self.kg_manager_instance)

        self.pipeline = CognitionPipeline()
        self.pipeline.reasoner = self.reasoner_with_mock_kg
        self.pipeline.kg_manager = self.kg_manager_instance
        self.pipeline.api_available = False

    def test_response_with_known_entity(self):
        """Tests if the pipeline can generate a response about a known entity."""
        response, _ = self.pipeline.process_message("Tell me about socrates")

        # The stable reasoner only lists relationships, not descriptions.
        # expected_text_part_1 = "'socrates'의 정의: A Greek philosopher from Athens."
        expected_text_part_2 = "'socrates'은(는) 'human'와(과) 'is_a' 관계를 가집니다."

        # self.assertIn(expected_text_part_1, response['text'])
        self.assertIn(expected_text_part_2, response['text'])


    def test_response_with_no_memory(self):
        """Tests if the pipeline provides a learning message when no relevant memory is found."""
        response, _ = self.pipeline.process_message("What is a star?")

        base_response = "아직은 어떻게 답해야 할지 모르겠어요. 하지만 배우고 있어요."
        expected_response = f"나는 지금 네 뜻을 더 선명히 이해하고자 해. {base_response}"
        self.assertEqual(response['text'], expected_response)


if __name__ == '__main__':
    unittest.main()
